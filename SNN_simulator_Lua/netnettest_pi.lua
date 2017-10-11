-- See onepi.cfg for a sample configuration
-- Pi side version of netnettest

-- This hould wait for keyboard input, take a snapshot and forward it

--require 'libjpeg'

require 'networknetwork'
require 'gnuplot'
require 'sortlayer'
local utils = require 'utils'
local resmets = require 'resamplers'
local augment = require 'augment'
local image = require 'image'
local signal = require 'posix.signal'

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
nwnw:training()
nwnw:evaluate()

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

local picfile = 'temp.jpg'

local w, h = 224, 224

os.execute('v4l2-ctl --set-ctrl horizontal_flip=1')
os.execute('v4l2-ctl --set-ctrl vertical_flip=1')
os.execute('v4l2-ctl --set-fmt-video=width=' .. w .. ',height=' .. h .. ',pixelformat=2')

local pic = torch.zeros(3,w,h)

local function take_picture()
  local pipe = io.popen('v4l2-ctl --stream-mmap=3 --stream-count=1 --stream-to=-')
  local content = pipe:read('*a')
  local index = 1

  for y = 1, w do
    for x = 1, h do
      for c = 1, 3 do
        pic[c][y][x] = string.byte(content, index)
        index = index + 1
      end
    end
  end
end

print('ready')

while loop do
  local data = getData()

  if utils.kbhit() then
    print('key pressed')
    io.read()
--    os.execute('v4l2-ctl --stream-mmap=3 --stream-count=1 --stream-to=' .. picfile)
    print('pic taken')
    take_picture() --image.load(picfile)--image.load('/dev/stdin')
    print('pic loaded')
    --print(pic:size())
    data = 0.2126 * pic[1] + 0.7152 * pic[2] + 0.0722 * pic[3] --resmets.nn(pic, 28, 28) --augment.zca({resmets.nn(pic, 28, 28),{}})[1]
    print('enter pressed')
  end

  local output

  if data and (data:dim() == 3 or data:dim() == 2) then
    local mean, std = torch.mean(data), torch.std(data)
print(mean,std)
gnuplot.imagesc(data)
    data = (data - mean) / std
    if data:dim() == 2 then
      data = data:reshape(1, data:size(1), data:size(2))
    end
    data = torch.repeatTensor(data, 2, 1, 1, 1) -- Because logic
--    print(data:size())
  end

  loop, output = nwnw:step(data)
  if output then
--    print(output)
  end
  collectgarbage()
  collectgarbage()
end
