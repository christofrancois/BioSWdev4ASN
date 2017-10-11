--[[
This program trains neural networks given their topology and and training data.
It does not use validation (at least for now) and does not test the network.
Despite its name, it can also receive local data/network definitions and save them locally,
however it has been developed for use as a remote trainer to be given the network and data
remotely.

When given a local network, the program does not start a server, but just trains that network
with the given data and exits.

When used as a server, the output option is ignored
]]

require 'nn'
require 'optim'
require 'gnuplot'
require 'unsup'
require 'LargeMarginCriterion'

local socket = require 'socket'
local gl = require 'greedylayerwise'
local utils = require 'utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training of Deep Neural networks locally or over the network')
cmd:text('Examples:')
cmd:text('Train a local network')
cmd:text('$> th trainingserver.lua --network network.lua --data data.th7')
cmd:text('Start a training server')
cmd:text('$> th trainingserver.lua --server --address 127.0.0.1 --port 9898')
cmd:text('The IP address must be your own')
cmd:text('Train a network remotely')
cmd:text('$> th trainingserver.lua --network network.lua --data data.th7 --address 127.0.0.1 --port 9898')
cmd:text()
cmd:text('Options:')
cmd:option('--dataset', '', 'which dataset to use. Must be a torch data file.')
cmd:option('--network', '', 'network and settings definition file to use')
cmd:option('--output', '', 'output file to produce')
cmd:option('--address', '', 'IP address for accepting incoming connections')
cmd:option('--localaddress', '', 'local IP address when sending data to server')
cmd:option('--port', '', 'listening port')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:option('--server', false, 'run a network training server')
cmd:option('--select', 0, 'select and run only one config from a hyperparameter set')
cmd:text('if either network or dataset is defined, the other must too')
cmd:text()
opt = cmd:parse(arg or {})
local verbose
if opt.silent then
  verbose = function() end
else
  verbose = print
end

local function nillize(x)
  return x ~= '' and x or nil
end

opt.dataset = nillize(opt.dataset)
opt.network = nillize(opt.network)
opt.output = nillize(opt.output)
opt.address = nillize(opt.address)
opt.localaddress = nillize(opt.localaddress)
opt.port = tonumber(nillize(opt.port))

-- Never mind this...
local function sendData(data, connection, description)
  description = description or 'Sending'
  local serial = torch.serialize(data, 'ascii')
  local length = dataToSend:len()

  local bytesSent, errMsg = client:send(length .. '\n')

  if bytesSent then
    local chunkSize
    chunkSize, errMsg = client:receive()

    if errMsg then
      verbose(description .. ' failed: ' .. errMsg)
      return
    elseif chunkSize <= 0 then
      chunkSize = length
    end

    verbose( 'Sending ' .. utils.humanReadable(length)
          .. ' in chunks of ' .. utils.humanReadable(chunkSize))

    bytesSent = 0
    while bytesSent < length do
      local endPoint = bytesSent + chunkSize
      bytesSent, errMsg = client:send(serial, bytesSent, endPoint)
      if not bytesSent then
        verbose(description .. ' failed: ' .. errMsg)
        return
      end
    end

    if bytesSent then
      verbose('Sent ' .. utils.humanReadable(bytesSent))
    end
  end

  if not bytesSent then
    verbose(description .. ' failed: ' .. errMsg)
  end
end

local function recvData(socket, chunkSize)
  -- Receive required data
  local bytes, errMsg = socket:receive()

  bytes = tonumber(bytes)

  if bytes then
    verbose('Receiving ' .. utils.humanReadable(bytes) .. ' of data')
  else
    verbose('Error receiving data: ' .. errMsg)
  end

  local dataLeft = bytes

  return function()
    if dataLeft <= 0 then
      return nil
    end
--print(math.min(chunkSize, dataLeft))
    local chunk, errMsg = socket:receive(math.min(chunkSize, dataLeft))

    if not chunk then
      error('Error receiving data: ' .. errMsg)
    end

    dataLeft = dataLeft - chunkSize
    return chunk
  end
end

verbose(opt)

local timeout = 120

local function selectConfig(configs, n)
  if type(configs) == 'table' then
    return configs[n]
  else
    for i, conf in configs do
      if i == n then
        return conf
      end
    end
  end
  error('Config could not be selected!')
end

local function train(settings, silent)
  if type(settings.network.trainset) == 'function' then
    verbose('Transforming datset')
    settings.network.trainset = settings.network.trainset(settings.network.dataset)
  end

  if not settings.network.modules and not settings.network.configs then
    return settings.network.trainset
  end

  local trained, valAcc
  if settings.network.select then opt.select = settings.network.select end

  if    settings.network.testSet
    and settings.network.validator
    and (not settings.network.configs or opt.select > 0) then
    if settings.network.trainset.mean then
      settings.network.testSet[1] = (settings.network.testSet[1] - settings.network.trainset.mean) / settings.network.trainset.std
    end
    local validator = settings.network.validator
    settings.network.validator = function(x) return validator(x, settings.network.testSet) end
  end

  if settings.network.configs then
    if opt.select <= 0 then
      trained = gl.hyperGreedyLayerwise(settings.network, silent)
    else
      local config = selectConfig(settings.network.configs, opt.select)
      config.trainset = #config.trainset > 0 and config.trainset or settings.network.trainset
      if settings.network.validator then
        config.validator = settings.network.validator
      end
      trained, valAcc = gl.greedyLayerwise(config, silent)
      settings.network.epochs = config.epochs
    end
  else
    if type(settings.network.method) == 'string' then
      settings.network.method = utils.getByName(settings.network.method)
    end

    trained, valAcc = gl.greedyLayerwise(settings.network, silent)
  end

  if valAcc then
    gnuplot.figure()
    gnuplot.plot('Validation accuracy', torch.range(1, settings.network.epochs), valAcc, '-')
  end

  return trained
end

if opt.network or opt.dataset then
  -- Non-server Mode
  local dataset = nil

  if opt.dataset then
    verbose('Loading dataset')
    dataset = assert(torch.load(opt.dataset, 'ascii'), 'data loading failed')
  end

  verbose('Loading network')
  local network = assert(dofile(opt.network), 'loading the network failed')

  if dataset then
    if type(network.trainset) == 'function' then
--      verbose('Transforming dataset')
--      network.trainset = network.trainset(dataset)
      network.dataset = dataset
    else
      network.trainset = dataset
    end
  else
    assert(network.trainset, 'no assigned training data and no default training data')
  end

  local outputfile = opt.output or opt.network .. '.out'

  local file = assert(io.open(outputfile, 'w'), 'opening the output file failed')
  local trained

  if opt.address and opt.port then
    verbose('Training remotely @ ' .. opt.address .. ':' .. opt.port)
    local message = torch.serialize({ network = network }, 'ascii')
    local client, errMsg = socket.connect(opt.address, opt.port, opt.localaddress)

    if not client then
      error('Connection failed: ' .. errMsg)
    end

    client:settimeout(timeout)
    verbose(message:len())
    local bytesSent, errMsg = client:send(message:len() .. '\n')

    if bytesSent then
      bytesSent, errMsg = client:send(message)
    end

    if not bytesSent then
      verbose('Sending network failed: ' .. errMsg)
      return
    end

    client:settimeout(nil)
--[[    local nBytes
    nBytes, errMsg = client:receive()

    if nBytes then
      verbose('Receiving ' .. nBytes .. ' bytes')
      trained, errMsg = client:receive(nBytes)
    end

    if not trained then
      verbose('Receiving trained network failed: ' .. errMsg)
      return
    else
      trained = torch.deserialize(trained, 'ascii')
    end]]

    for chunk in recvData(client, 1024*1024) do
      file:write(chunk)
    end

    client:close()
  else
    verbose('Training locally')
    --[[
    if type(network.method) == 'string' then
      network.method = utils.getByName(network.method)
    end
    trained = gl.greedyLayerwise(network, opt.silent)
    ]]
    trained = train({ network = network }, opt.silent)
    file:close()
    torch.save(outputfile, trained, 'ascii')
    --file:write(torch.serialize(trained, 'ascii'))
  end

  return
end

-- Server Mode

-- Setup a listening port
local address = assert(opt.address, 'no IP address specified for server')
local port = assert(opt.port, 'no port specified for server')
local server = socket.bind(address, port)
server:settimeout(5)

while true do -- Listen to requests indefinitely (quit using ^C)
  local client = server:accept() -- Accept connection from remote client

  if client then
    client:settimeout(timeout) -- The client has 5 seconds to initiate data transfer

    -- Receive required data
    local bytes, errMsg = client:receive()
    local settings

    if bytes then
      verbose(bytes)
      settings, errMsg = client:receive(bytes)
    end

    if not settings then
      verbose('Error receiving data: ' .. errMsg)
    else
      -- This should be done inside pcall
      -- We want a table with the element 'network' as the network to be trained
      -- We may later want to allow other settings such as using a different training strategy
      settings = torch.deserialize(settings, 'ascii')
      verbose(settings)

      --if type(settings.network.method) == 'string' then
      --  settings.network.method = utils.getByName(settings.network.method)
      --end

      -- As should this
      --local trained = gl.greedyLayerwise(settings.network, opt.silent)
      local trained = train(settings, opt.silent)
      local dataToSend = torch.serialize(trained, 'ascii')

      local bytesSent
      bytesSent, errMsg = client:send(dataToSend:len() .. '\n')

      if bytesSent then
        verbose('Sending ' .. dataToSend:len() .. ' bytes')
        bytesSent, errMsg = client:send(dataToSend)
        if bytesSent then
          verbose('Sent ' .. bytesSent .. ' bytes')
        end
      end

      if not bytesSent then
        verbose('Sending trained network failed: ' .. errMsg)
      end
    end

    client:close()
  end
end
