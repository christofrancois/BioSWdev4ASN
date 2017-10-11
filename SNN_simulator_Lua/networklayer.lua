--[[
Defines NetworkInput and NetworkOutput modules for torch nn

requires the LuaSocket library

NetworkInput waits for input from network when forwarding and
passes gradOutput etc. to network when backwarding.
NetworkOutput does the opposite.

The NotworkOutput is always the client

These modules can be used to pass serializable data in general outside nn modules
]]

local socket = require 'socket'

-- Listening port
local upstreamPort = 24190

local NetworkInput = torch.class('nn.NetworkInput','nn.Module')
local NetworkOutput = torch.class('nn.NetworkOutput','nn.Module')

-- NetworkInput

function NetworkInput:__init(localaddress, remoteaddress, localport, timeout)
  localport = localport or upstreamPort
  self._server = socket.bind(localaddress, localport)
  self._server:settimeout(timeout)
  self._client = self._server:accept() -- Accept connection from remote NetworkOutput
  self._client:settimeout(timeout)
end

function NetworkInput:updateOutput()
  local input = self._client:receive()
  self.output = torch.deserialize(input, 'ascii')
  return self.output
end

function NetworkInput:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  self._client:send(torch.serialize(gradOutput, 'ascii'))
  return self.gradInput
end

-- NetworkOutput

function NetworkOutput:__init(localaddress, remoteaddress, remoteport, timeout)
  localport = localport or upstreamPort
  self._client = socket.connect(remoteaddress, remoteport, localaddress)
  self._client:settimeout(timeout)
end

function NetworkOutput:updateOutput(input, target)
  self.output = input
  self._client:send(torch.serialize(input, 'ascii'))
  return self.output
end

function NetworkOutput:updateGradInput(input)
  local gradOutput = self._client:receive()
  self.gradInput = torch.deserialize(gradOutput, 'ascii')
  return self.gradInput
end


