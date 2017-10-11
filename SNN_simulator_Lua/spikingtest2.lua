require 'spikingreservoir'
require 'gnuplot'
local models = require 'models'

local file = assert(io.open('data.dat', 'w'))

local i = 0
local timeStep = 0.001
local exciteSteps = 1000
local restSteps = 1000
local relearnSteps = 1000
local totalSteps = exciteSteps + restSteps + relearnSteps
local window = 200
local inputs = 28*28
local outputs = 28*28
local inspike = 0.15/timeStep
local infreq = 150

local sr = nn.SpikingReservoir
  { neurons = {models.default(2*28*28), models.fastSpiking(200)}
  , connectivity = 0.2
  , inputs = torch.range(1, inputs):long()
  , outputs = torch.range(inputs + 1, inputs + outputs):long()
  , spikeCallback = function(x, time) file:write(time .. ' ' .. x .. '\n'); i = i + 1 end
  , timeStep = timeStep
  , subSteps = 1
  }

local potentials = torch.zeros(totalSteps, sr._nTotal)
local spikes1 = torch.zeros(window, outputs)
local spikes2 = torch.zeros(window, outputs)
local spikes3 = torch.zeros(window, outputs)
local mnist = torch.load('/home/lehtolav/data/mnist/train.th7', 'ascii')
local firstpic = mnist[1][1]:reshape(28*28):double()

gnuplot.imagesc(firstpic:reshape(28,28))
gnuplot.figure()
gnuplot.imagesc(sr._weights ,'color')

print('running simulation')
local time1 = os.time()

for i = 1, exciteSteps, 1 do
  sr:forward(torch.lt(torch.rand(inputs), firstpic * infreq * timeStep):double() * inspike)
  sr._potential:indexAdd(1, sr._outputs, torch.lt(torch.rand(inputs), firstpic * infreq):double() * inspike)
  potentials[i] = sr._potential

  if i > exciteSteps - window then
    spikes1[i + window - exciteSteps] = sr._spikes:index(1, sr._outputs)
  end
end
for i = 1 + exciteSteps, exciteSteps + restSteps, 1 do
  sr:forward(torch.zeros(inputs))
  potentials[i] = sr._potential

  if i > exciteSteps + restSteps - window then
    spikes2[i + window - (exciteSteps + restSteps)] = sr._spikes:index(1, sr._outputs)
  end
end
for i = 1 + exciteSteps + restSteps, totalSteps, 1 do
  sr:forward(torch.lt(torch.rand(inputs), firstpic * infreq * timeStep):double() * inspike)
  potentials[i] = sr._potential

  if i > totalSteps - window then
    spikes3[i + window - totalSteps] = sr._spikes:index(1, sr._outputs)
  end
end

print('simulation ended')
print('ran ' .. (totalSteps) * sr._timeStep .. ' seconds of simulation in ' .. os.time() - time1 .. ' seconds')

file:flush()
file:close()

gnuplot.figure()
gnuplot.imagesc(sr._weights ,'color')
gnuplot.figure()
gnuplot.raw("plot 'data.dat' with dots")
gnuplot.figure()
gnuplot.imagesc(potentials:t() ,'color')

print('total ' .. i .. ' spikes')

local spikerates = torch.sum(spikes1, 1) / (window * sr._timeStep)
gnuplot.figure()
gnuplot.imagesc(spikerates:reshape(28,28) / 1000)
spikerates = torch.sum(spikes2, 1) / (window * sr._timeStep)
gnuplot.figure()
gnuplot.imagesc(spikerates:reshape(28,28) / 1000)
spikerates = torch.sum(spikes3, 1) / (window * sr._timeStep)
gnuplot.figure()
gnuplot.imagesc(spikerates:reshape(28,28) / 1000)
