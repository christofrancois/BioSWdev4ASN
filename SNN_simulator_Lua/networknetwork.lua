--[[
Uses trained networks over the network

Intended features:
- Gather data from several remote networks
- Server asks for data and overall controls the flow of the program
- Clients should be able to push data too (so that we can activate the network on clientside events)
- Make a graph like topology? (servers can also be clients)
- Interactivew training?
]]

require 'nn'
--require 'optim'
local socket = require 'socket'
local utils = require 'utils'
local xlua = xlua

if not xlua then
  xlua = require 'xlua'
end

--local newport = 49001 -- Start assigning port numbers for server connections from here

-- Requires a network connection settings file (local neighborhood)
-- Network to use

-- Network connection setting must define a datasource

-- A server sends a 'forward' message to push or pull data up the network
-- A client/server may send a 'backward' to push data down, but the server/client may refuse
-- Reasons for refusal include 'not in training' and 'currently forwarding'
-- Data must be tagged with a number to indicate order so that backward calls
--  can be associated with forwards

local logNetError = error
local debug = print
local sendTimeout = 0
local recvTimeout = 1

local function call(f, ...)
  local result, errMsg = f(...)

  if not result then
    logNetError('Network error: ' .. errMsg)
  end

  return result
end

local function packMessage(senderIP, packNumber, messageType, payload)
  if not senderIP or not packNumber then
    logNetError('senderIP or packNumber was nil!')
  end

  return senderIP .. '\n'
      .. packNumber .. '\n'
      .. messageType .. '\n'
      .. (payload and payload:len() or 0) .. '\n'
      .. (payload or '')
end

local function unpackMessage(socket)
  local sender, errMsg = socket:receive()
  if not sender then
    if errMsg ~= 'timeout' then
      logNetError('unpack failed with ' .. errMsg)
    end
  else
--    debug('getting packNumber')
    local packNumber = tonumber(call(socket.receive, socket))
--    debug(packNumber, 'getting messageType')

    local messageType = call(socket.receive, socket)
--    debug(messageType, 'getting length')

    local length = tonumber(call(socket.receive, socket))
--    debug(length, 'getting payload')

    local payload
    if length > 0 then
      payload = call(socket.receive, socket, length)
    end

--    debug(payload)
    return sender, packNumber, messageType, nil or (length > 0 and payload)
  end
end

-- Message types as variables to reduce impact of typos
local forward_request = 'forward_request'
local forward_data = 'forward_data'
local forward_top = 'forward_top'
local backward_data = 'backward_data'
local backward_ack = 'backward_ack'
local quit = 'quit'

local NetworkNetwork = torch.class('nn.NetworkNetwork', 'nn.Module')

function NetworkNetwork:__init(config)
  local args, incoming, outgoing, network, inCollector, outDistributor, packTimeout, idString, criterion, postProcess = xlua.unpack(
      {config},
      'NetworkNetwork', 
      'A network connected neural network',
      {arg='incoming', type='table',
       help='Table of incoming connections'},
      {arg='outgoing', type='table',
       help='Table of outgoing connections'},
      {arg='network', type='nn.Module',
       help='Local neural network to use'},
      {arg='inCollector', type='function',
       help='Compiles data from inputs to input for the local network'},
      {arg='outDistributor', type='function',
       help='Compiles local networks output into an array describing the recipient and data for them'},
      {arg='packTimeout', type='number',
       help='Time (in seconds) after which a packet will be considered failed when no response has been received'},
      {arg='idString', type='string',
       help='Unique identifier for this node. Typically the IP address.'},
      {arg='criterion', type='nn.Module', req=false,
       help='Whether or not to backward data after forwarding (or to just send ack). Only relevant to top node(s).'},
      {arg='postProcess', type='function', req=false,
       help='Function that postprocesses network output. Can for example reject output by outputting nil to prevent further forward propagation through the network.'}
  )
  --[[ Disregard the following
  -- Incoming means we are the client / outgoing that we are the server
  -- Servers can pull data with 'forward' if the outgoing connection is defined an input
  -- i.e. outgoing is not necessarily output and incoming not necessarily input
  ]]

  -- Outputs should not be differentiated to incoming and outgoing

  -- Id is a pair of data with IP (or other identifier) of the initiator of the message
  -- and the sequential number of the event in the network
  -- This way the original messager can know if the event has been acknowledged or if the
  -- incoming event, with the event number they chose is for a different event
  -- All messages contain id
  -- Possible messages
  -- 'forward request':
  --   request a forwading in the network. propagated downwards in the net until reaches bottom
  --   data is then generated and propagated upward
  -- 'forward data':
  --   output data of a below layer being forwarded upward
  -- 'forward top':
  --   request for the top layer to propagate a forward request
  -- 'backward data':
  --   gradient output data from an above layer
  -- 'backward ack':
  --   acknowledges a successful forward, without providing gradient data
  --   allows nodes to remove data about the previous forward request of the same id
  -- 'quit':
  --   shuts down all participant nodes

  self._incoming = incoming -- input connections
  self._outgoing = outgoing -- output connections
  self._connections = utils.tableConcat(incoming, outgoing) -- all connections
  self._network = network
  self._inCollector = inCollector
  self._outDistributor = outDistributor
--  self._incomingInputs = {}
--  self._outgoingInputs = {}
--  self._inputs = {}
--  self._outputs = {}
  self._packNumber = 0 -- Number for labeling new packs
  --self._prevData = -1 -- packNumber of the previous data forwarded/backwarded throught the network
  -- for generality we allow asynchronous forwarding, i.e. allow new forwards before a backward
  -- However if you want to train the network over the network, you will need synchrony
  self._packInfo = {}
  self._nIncoming = 0
  self._nOutgoing = 0
  self._packTimeout = packTimeout
  self._loop = true -- True until we want to quit
  self._idString = idString
  self._criterion = criterion
  self._postProcess = postProcess or function(x) return x end

  for _, v in pairs(self._incoming) do
    v.server = call(socket.bind, v.localAddress, v.localPort)
    v.socket = call(v.server.accept, v.server)
    v.peer = call(v.socket.getpeername, v.socket)
    v.socket:settimeout(recvTimeout)
    self._nIncoming = self._nIncoming + 1
--    if v.input then
--      self._incomingInputs[v.peer] = v
--    else
--      self._outputs[v.peer] = v
--    end
  end

  for _, v in pairs(self._outgoing) do
    v.socket = call(socket.connect, v.remoteAddress, v.remotePort, v.localAddress, v.localPort)
    v.peer = v.remoteAddress
    v.socket:settimeout(recvTimeout)
    self._nOutgoing = self._nOutgoing + 1
--    if v.input then
--      self._outgoingInputs[v.peer] = v
--    else
--      self._outputs[v.peer] = v
--    end
  end
end

function NetworkNetwork:step(input)
--[[
  for _, v in pairs(self._incomingInputs) do
    local length = call(v.socket.receive, v.socket)
    local data = call(v.socket.receive, v.socket, length)

    self._inputs[v.peer] = data
    self._packNumber = data.n > self._packNumber and data.n or self._packNumber
  end

  for _, v in pairs(self._outgoingInputs) do
    local length = call(v.socket.receive, v.socket)
    local data = call(v.socket.receive, v.socket, length)

    self._inputs[v.peer] = data
    self._packNumber = data.n > self._packNumber and data.n or self._packNumber
  end]]
  local output

  for _, v in pairs(self._connections) do
    local originIP, packNumber, messageType, payload = unpackMessage(v.socket)

    if originIP then
      print(originIP, packNumber, messageType)
    end

    local messageHandler = self[messageType]
  
    if messageHandler then
      messageHandler(self, v.peer, originIP, packNumber, payload)
    elseif messageType then
      logNetError('Unknown message type: ' .. messageType)
    end
  
    -- self._packNumber is used for new packs so they should have a number that
    -- is not yet used in the network, so we should update our packNumber to
    -- to the first free number (as far as we know)
    if packNumber and packNumber >= self._packNumber then
      self._packNumber = packNumber + 1
    end
  
    if input and not self._forwardedNumber then
--      print(input:size())
      output = self._postProcess(self._network:forward(input))
      if output then
        self._forwardedNumber = self._packNumber
        self._packInfo[self._packNumber] =
          { messageType = forward_data
          , originIP = self._idString
          }
        self._packNumber = self._packNumber + 1
        self:sendUp(self._idString, self._forwardedNumber, forward_data, torch.serialize(output, 'ascii'))
      end
    end

    for k, v in pairs(self._packInfo) do
      if v.messageType == forward_data and not self._forwardedNumber then
        if utils.countEntries(v.inputs) == self._nIncoming then
          local input = self._inCollector(v.inputs)
          debug(input)
          if input then
            output = self._postProcess(self._network:forward(input))
            if self._nOutgoing == 0 or not output then -- Top node of hierarchy or output rejected
              if self._criterion then
                -- TODO implement? would require a target
                -- and including an optimizer i guess if we want to actually learn
                error('Backwarding data not implemented')
              else
                self:sendDown(v.originIP, k, backward_ack)
              end
              self._packInfo[k] = nil
            else -- not top node
              self._forwardedNumber = k
              self:sendUp(v.originIP, k, forward_data, torch.serialize(output, 'ascii'))
            end
          end
        end
      elseif v.messageType == backward_data and self._forwardedNumber == k then
        if utils.countEntries(v.outputs) == self._nOutgoing then
          local input = self._outDistributor(v.inputs)
          if input then
            output = self._network:backward(input)
            self._forwardedNumber = nil
            self:sendDown(v.originIP, k, backward_data, torch.serialize(output, 'ascii'))
            self._packInfo[k] = nil
          else
            self._forwardedNumber = nil
            self:sendDown(v.originIP, k, backward_ack)
            self._packInfo[k] = nil
          end
        end
      end

      if not v.timeStamp then
        v.timeStamp = os.clock()
      elseif v.timeStamp - os.clock() > self._packTimeout then --outdated
        self._packInfo[k] = nil
      end
    end
  end

  return self._loop, output
end

NetworkNetwork[forward_request] = function(self, senderIP, originIP, packNumber, payload)
  local packInfo = self._packInfo[packNumber]

  if not packInfo then
    self._packInfo[packNumber] =
      { messageType = forward_data
      , originIP = originIP
      }
    self:sendDown(originIP, packNumber, forward_request)
  end
  -- Requests do not override other data
  -- Later this might (and should) generate a failure message so that the
  -- originator of the request can re-request
end

NetworkNetwork[forward_data] = function(self, senderIP, originIP, packNumber, payload)
  local packInfo = self._packInfo[packNumber]

  if packInfo then
    if packInfo.messageType == forward_request then
      if originIP ~= packInfo.originIP then
        -- Request collision - data takes precedence
        -- Request must have originated from below this node (from one branch)
        -- Send new requests for data down (for other branches)
        self:sendDown(originIP, packNumber, forward_request)
      end
      local input = torch.deserialize(payload, 'ascii')
      self._packInfo[packNumber] =
        { messageType = forward_data
        , inputs = { [senderIP] = input }
        , originIP = originIP
        }
    elseif packInfo.messageType == forward_data then
      if originIP == packInfo.originIP then
        -- Additional data for previous forward request
        local input = torch.deserialize(payload, 'ascii')
        self._packInfo[packNumber].inputs[senderIP] = input
      end
      -- Previous data takes precedence with conflicting data
      -- Do nothing and just drop the new data
    else -- Other previous data
      -- Override previous data
      -- This should happen though, since only input/gradient is stored normally and
      -- gradients should not be stored until after a successful forward
      -- (i.e. the forward has propagated through the network and reached the top)
      local input = torch.deserialize(payload, 'ascii')
      self._packInfo[packNumber] =
        { messageType = forward_data
        , inputs = { [senderIP] = input }
        , originIP = originIP
        }
    end
  else
    local input = torch.deserialize(payload, 'ascii')
    self._packInfo[packNumber] =
      { messageType = forward_data
      , inputs = { [senderIP] = input }
      , originIP = originIP
      }
  end
end

NetworkNetwork[forward_top] = function(self, senderIP, originIP, packNumber, payload)
  local packInfo = self._packInfo[packNumber]

  if not packInfo then
    self._packInfo[packNumber] =
      { messageType = forward_top
      , originIP = originIP
      }
    self:sendUp(originIP, packNumber, forward_top)
  end
  -- Requests do not override other data
  -- Later this might (and should) generate a failure message so that the
  -- originator of the request can re-request
end

NetworkNetwork[backward_data] = function(self, senderIP, originIP, packNumber, payload)
  local packInfo = self._packInfo[packNumber]

  if packInfo then
    -- There should be previous info for this
    -- The only legal reason for not having any is that it has timed out
    local input = torch.deserialize(payload, 'ascii')
    if packInfo.originIP ~= originIP then
      logNetError('backward data originator mismatch')
    elseif self._forwardedNumber ~= packNumber then
      logNetError('backward data for non-current forward')
    elseif packInfo.messageType == forward_data then
      self._packInfo[packNumber] =
        { messageType = backward_data
        , inputs = { [senderIP] = input }
        , originIP = originIP
        }
    elseif packInfo.messageType == backward_data then
      self._packInfo[packNumber].inputs[senderIP] = input
    else
      -- This should never happen
      logNetError('Unexpected message sequence: '
        .. packInfo.messageType .. ' data before ' .. backward_data)
      --self._packInfo[packNumber] = nil
    end
  else
  -- We drop the message if there is no previous data
  end
end

NetworkNetwork[backward_ack] = function(self, senderIP, originIP, packNumber, payload)
  local packInfo = self._packInfo[packNumber]

  if packInfo then
    if packInfo.originIP ~= originIP then
      logNetError('backward ack originator mismatch')
    elseif packInfo.messageType == forward_data then
      self._packInfo[packNumber] = nil
      self._forwardedNumber = nil
      self:sendDown(senderIP, packNumber, backward_ack)
    else
      logNetError('Unexpected message sequence: '
        .. packInfo.messageType .. ' data before ' .. backward_ack)
    end
  end
end

NetworkNetwork[quit] = function(self, senderIP, originIP, packNumber, payload)
  self._loop = false
  self:sendUp(originIP, packNumber, quit)
  self:sendDown(originIP, packNumber, quit)
  for k, v in pairs(self._connections) do
    v.socket:close()
  end
end

function NetworkNetwork:sendUp(senderIP, packNumber, messageType, payload)
  for _, v in pairs(self._outgoing) do
    v.socket:settimeout(sendTimeout)
    local packed = packMessage(senderIP, packNumber, messageType, payload)
    call(v.socket.send, v.socket, packed)
    v.socket:settimeout(recvTimeout)
  end
end

function NetworkNetwork:sendDown(senderIP, packNumber, messageType, payload)
  for _, v in pairs(self._incoming) do
    v.socket:settimeout(sendTimeout)
    call(v.socket.send, v.socket, packMessage(senderIP, packNumber, messageType, payload))
    v.socket:settimeout(recvTimeout)
  end
end

function NetworkNetwork:evaluate()
  self._network:evaluate()
end

function NetworkNetwork:training()
  self._network:training()
end
