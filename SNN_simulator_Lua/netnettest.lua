-- See onepi.cfg for a sample configuration

require 'networknetwork'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Using neural networks over the network')
cmd:text('Options:')
cmd:option('--dataset', '', 'which dataset to use. A lua table of samples.')
cmd:option('--network', '', 'stored network to use')
cmd:option('--config', '', 'configuration for network node')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})

local verbose
if opt.silent then
  verbose = function() end
else
  verbose = print
end

for k, v in pairs(opt) do
  if v == '' then
    opt[k] = nil
  end
end

if not opt.config then
  error('A config file is needed!')
elseif not opt.network then
  error('A trained network is needed!')
end

local config, errMsg = dofile(opt.config)
if not config then
  error(errMsg)
end

local network
network, errMsg = torch.load(opt.network, 'ascii')
if not config then
  error(errMsg)
end

config.network = network
local nwnw = nn.NetworkNetwork(config)

local getData = config.dataSource

if opt.dataset then
  local dataset = torch.load(opt.dataset, 'ascii')
  local data = next(dataset, nil)
  getData = function()
    if data then
      local temp = data
      data = next(dataset, data)
      return temp
    end
    return config.dataSource()
  end
end

local loop = true

local heebot = {'ville', 'frank', 'timo', 'mikko'}

while loop do
  local data = getData()
  local output
  loop, output = nwnw:step(data)
  if output then
    print(output)
    local _, i = torch.max(output:squeeze(), 1)
    print(i)
    print(heebot[i:squeeze()])
  end
end
