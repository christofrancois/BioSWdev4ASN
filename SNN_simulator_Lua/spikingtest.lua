require 'spikingreservoir'
require 'gnuplot'
local models = require 'models'

local file = assert(io.open('data.dat', 'w'))

local i = 0
local timeStep = 0.001
local window = 100
local inputs = 28*28
local outputs = 10
local inspike = 0.15/timeStep
local infreq = 150

local sr = nn.SpikingReservoir
  { neurons = {models.default(inputs + outputs + 200), models.fastSpiking(300)}
  , connectivity = 0.2
  , inputs = torch.range(1, inputs):long()
  , outputs = torch.range(inputs + 1, inputs + outputs):long()
  , spikeCallback = function(x, time) file:write(time .. ' ' .. x .. '\n'); i = i + 1 end
  , timeStep = timeStep
  , subSteps = 1
  }

--local potentials = torch.zeros(totalSteps, sr._nTotal)
local spikes = torch.zeros(window, outputs)
local mnist = torch.load('/home/lehtolav/data/mnist/train.th7', 'ascii')
local pics = 50
local stepsPerPic = 100
local restSteps = 50
local testPics = 50
local stepsPerTest = 100

gnuplot.imagesc(sr._weights ,'color')

print('running simulation')
local time1 = os.time()
local j= 0

for i = 1, pics, 1 do
  print('training picture ' .. i)
  local pic = mnist[1][i]:reshape(28*28):double()
  local target = torch.zeros(outputs)
  target[mnist[2][i] + 1] = 1
  for j = 1, stepsPerPic, 1 do
    sr:forward(torch.lt(torch.rand(inputs), pic * infreq * timeStep):double() * inspike)
    sr._potential:indexAdd(1, sr._outputs, torch.lt(torch.rand(outputs), target * infreq):double() * inspike)
  end

  for j = 1, restSteps, 1 do
    sr:forward(torch.zeros(inputs))
  end
end

local corrects = 0

for i = 1, testPics, 1 do
  print('testing picture ' .. i)
  local pic = mnist[1][pics + i]:reshape(28*28):double()
  local target = mnist[2][pics + i]
  for j = 1, stepsPerTest, 1 do
    sr:forward(torch.lt(torch.rand(inputs), pic * infreq * timeStep):double() * inspike)
    if j > stepsPerTest - window then
      spikes[j + window - stepsPerTest] = sr._spikes:index(1, sr._outputs)
    end
  end
  local spikerates = torch.sum(spikes, 1) / (window * sr._timeStep)
  local y, ind = torch.max(spikerates, 2)
  ind = ind:squeeze()
  print('predicted ' .. ind .. ', correct was ' .. target)
  print(y:squeeze())
  print(spikerates)
  if ind == target then
    corrects = corrects + 1
  end

  for j = 1, restSteps, 1 do
    sr:forward(torch.zeros(inputs))
  end
end

print('simulation ended')
print('ran ' .. sr._time .. ' seconds of simulation in ' .. os.time() - time1 .. ' seconds')

file:flush()
file:close()

gnuplot.figure()
gnuplot.imagesc(sr._weights ,'color')
gnuplot.figure()
gnuplot.raw("plot 'data.dat' with dots")

print('total ' .. i .. ' spikes')
print('prediction rate ' .. corrects / testPics)
--[[
local spikerates = torch.sum(spikes1, 1) / (window * sr._timeStep)
gnuplot.figure()
gnuplot.imagesc(spikerates:reshape(28,28) / 1000)
spikerates = torch.sum(spikes2, 1) / (window * sr._timeStep)
gnuplot.figure()
gnuplot.imagesc(spikerates:reshape(28,28) / 1000)
spikerates = torch.sum(spikes3, 1) / (window * sr._timeStep)
gnuplot.figure()
gnuplot.imagesc(spikerates:reshape(28,28) / 1000)]]
