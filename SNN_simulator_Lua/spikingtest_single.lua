require 'spikingreservoir_rk'
require 'gnuplot'
local models = require 'models'
local numint = require 'numint'

local file = assert(io.open('data.dat', 'w'))

local timeScale = 5
local spikeN = 0
local timeStep = 1e-5*timeScale
local testLength = 12000
local exciteSteps = 40000/timeScale--math.floor(testLength / 2 + 0.5)
local restSteps = 0--math.floor(testLength / 6 + 0.5)
local relearnSteps = 0-- math.floor(testLength / 6 * 2 + 0.5)
local totalSteps = exciteSteps + restSteps + relearnSteps
local window = 200
local inputs = 1
local outputs = 1
local inspike = 55/timeStep
local infreq = 100
local printEvery = 100

local sr = nn.SpikingReservoir
  { neurons = {models.regularSpiking(1,{
      C = models.constant(100)
    , k = models.constant(0.7)
    , v_r = models.constant(-60)
    , v_t = models.constant(-40)
    , v_peak = models.constant(35)
    , a = models.constant(0.03)
    , b = models.constant(-2)
    , c = models.constant(-50)
    , d = models.constant(400)
    , tau_x = models.constant(100)
    , p = models.constant(0.6)
    , g_AMPA = models.constant(1)
    , g_NMDA = models.constant(1)
    , g_GABAA = models.constant(0)
    , g_GABAB = models.constant(0)
    })}
  , connectivity = 0.2
  , inputs = torch.range(1, 1):long()
  , outputs = torch.range(1, 1):long()
  , spikeCallback = function(x, time) file:write(time .. ' ' .. x .. '\n'); spikeN = spikeN + 1 end
  , timeStep = timeStep
  , subSteps = 1
  , n = 1
  , method = numint.RK4
  }

local potentials = torch.zeros(totalSteps, sr._nTotal)
local spikes1 = torch.zeros(window, outputs)
local spikes2 = torch.zeros(window, outputs)
local spikes3 = torch.zeros(window, outputs)
local monitored = 1
local monitorVars = {'_potential', '_recovery'}
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

print('running simulation')
local time1 = os.time()

for i = 1, exciteSteps, 1 do
  sr:forward(torch.ones(inputs)*55)--torch.lt(torch.rand(inputs), infreq * timeStep):double() * inspike)
  potentials[i] = sr._potential

  if i > exciteSteps - window then
    spikes1[i + window - exciteSteps] = sr._spikes:index(1, sr._outputs)
  end
  updMonitors(i)
  if i % printEvery == 0 then print(i .. '/' .. totalSteps .. ', ' .. spikeN .. ' spikes') end
end
for i = 1 + exciteSteps, exciteSteps + restSteps, 1 do
  sr:forward(torch.zeros(inputs))
  potentials[i] = sr._potential

  if i > exciteSteps + restSteps - window then
    spikes2[i + window - (exciteSteps + restSteps)] = sr._spikes:index(1, sr._outputs)
  end
  updMonitors(i)
  if i % printEvery == 0 then print(i .. '/' .. totalSteps .. ', ' .. spikeN .. ' spikes') end
end
for i = 1 + exciteSteps + restSteps, totalSteps, 1 do
  sr:forward(torch.lt(torch.rand(inputs), infreq * timeStep):double() * inspike)
  potentials[i] = sr._potential

  if i > totalSteps - window then
    spikes3[i + window - totalSteps] = sr._spikes:index(1, sr._outputs)
  end
  updMonitors(i)
  if i % printEvery == 0 then print(i .. '/' .. totalSteps .. ', ' .. spikeN .. ' spikes') end
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
