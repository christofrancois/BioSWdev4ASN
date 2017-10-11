require 'spikingreservoir_rk'
require 'gnuplot'
local models = require 'models'

local file = assert(io.open('data.dat', 'w'))

local spikeN = 0
local timeStep = 0.001 --s
local learnLength = 350 --ms
local restLength = 150 --ms
local trainImages = 5
local totalSteps = trainImages * (learnLength + restLength) * 1000 * timeStep
local window = 200
local inputs = 28*28
local inputNeurons = 300
local outputs = 25
local inspike = 1/timeStep
local infreq = 150
local printEvery = 100
local currentStep = 1

local sr = nn.SpikingReservoir
  { neurons = { models.regularSpiking(800), models.fastSpiking(200)}
  , connectivity = 0.2
  , inputs = (torch.rand(inputs) * inputNeurons + 1):long()
  , outputs = torch.range(inputNeurons + 1, inputNeurons + outputs):long()
  , spikeCallback = function(x, time) file:write(time .. ' ' .. x .. '\n'); spikeN = spikeN + 1 end
  , timeStep = timeStep
  , subSteps = 1
  , n = 1
  , method = numint.RK4
  }

local potentials = torch.zeros(totalSteps, sr._nTotal)
local spikes = torch.zeros(window, outputs)
local spikes2 = torch.zeros(window, outputs)
local spikes3 = torch.zeros(window, outputs)
local monitored = inputs + 1
local monitorVars = {'_potential', '_gAMPA', '_gNMDA', '_gGABAA', '_gGABAB', '_recovery'}
local monitors = {}

for _, v in pairs(monitorVars) do
  monitors[v] = torch.zeros(totalSteps)
end

local function updMonitors(i)
  for _, v in pairs(monitorVars) do
    monitors[v][i] = sr[v][monitored]
  end
end

gnuplot.imagesc(sr._weights ,'color')

local mnist_train = torch.load('/home/lehtolav/data/mnist/train.th7', 'ascii')

print('running simulation')
local time1 = os.time()

local function handleStep(input)
  sr:forward(input)
  potentials[currentStep] = sr._potential

  if currentStep > totalSteps - window then
    spikes[currentStep + window - totalSteps] = sr._spikes:index(1, sr._outputs)
  end

  updMonitors(currentStep)

  if currentStep % printEvery == 0 then
    print(currentStep .. '/' .. totalSteps .. ', ' .. spikeN .. ' spikes')
  end

  currentStep = currentStep + 1
end

for i = 1, trainImages do
  print('picture ' .. i)

  -- Ecxitational perioid
  print('learning for ' .. math.floor(learnLength * 0.001 / timeStep + 0.5) .. ' steps')
  local pic = mnist_train[1][i]:reshape(28*28):double()
  for j = 1, math.floor(learnLength * 0.001 / timeStep + 0.5) do
    handleStep(torch.lt(torch.rand(inputs), pic * infreq * timeStep):double() * inspike)
  end

  -- Rest perioid
  print('resting for ' .. math.floor(restLength * 0.001 / timeStep + 0.5) .. ' steps')
  for j = 1, math.floor(restLength * 0.001 / timeStep + 0.5) do
    handleStep(torch.zeros(inputs))
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

print('total ' .. spikeN .. ' spikes')

--[[
local spikerates = torch.sum(spikes1, 1) / (window * sr._timeStep)
gnuplot.figure()
gnuplot.imagesc(spikerates:reshape(5,5) / 1000)
spikerates = torch.sum(spikes2, 1) / (window * sr._timeStep)
gnuplot.figure()
gnuplot.imagesc(spikerates:reshape(5,5) / 1000)
spikerates = torch.sum(spikes3, 1) / (window * sr._timeStep)
gnuplot.figure()
gnuplot.imagesc(spikerates:reshape(5,5) / 1000)
]]

for _, v in pairs(monitorVars) do
  gnuplot.figure()
--  gnuplot.raw('title ' .. v)
  gnuplot.plot(v, torch.range(0, totalSteps - 1) * timeStep, monitors[v], '-')
end
