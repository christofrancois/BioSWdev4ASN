

package.path = package.path .. ';?.lua'

require 'spikingreservoir_rk'
require 'gnuplot'
local models = require 'models'
local numint = require 'numint'

local file = assert(io.open('data.dat', 'w'))

local spikeN = 0
local timeScale = 1/4
local timeStep = 1e-3*timeScale -- s / step
local testLength = 1 -- s
local exciteSteps = testLength / timeStep--math.floor(testLength / 2 + 0.5)
local restSteps = 0--math.floor(testLength / 6 + 0.5)
local relearnSteps = 0-- math.floor(testLength / 6 * 2 + 0.5)
local totalSteps = exciteSteps + restSteps + relearnSteps
local window = 200
local networkScale = 0.3
local inputs = 150 * networkScale
local outputs = 1
local inspike = 1--/timeStep
local infreq = (1/40)/timeStep
local printEvery = 1000

local sr = nn.SpikingReservoir
  { neurons = {models.regularSpiking(700*networkScale,{
      C = models.constant(100)
    , k = models.constant(0.7)--3)--0.7)
    , v_r = models.constant(-60)
    , v_t = models.constant(-40)
    , v_peak = models.constant(35)
    , a = models.constant(0.13)--0.03)--0.13)
    , b = models.constant(-3.5)
    , c = models.constant(-50)
    , d = models.constant(400)
    , tau_x = models.constant(100)
    , p = models.constant(0.65)
    , g_AMPA = models.constant(1)
    , g_NMDA = models.constant(1)
    , g_GABAA = models.constant(0)
    , g_GABAB = models.constant(0)
    }),models.fastSpiking(260*networkScale), models.latentSpiking(40*networkScale)}
  , connectivity = 0.02
  , inputs = torch.range(1, inputs):long()
  , outputs = torch.range(1 + inputs, inputs + outputs):long()
  , spikeCallback = function(x, time) file:write(time .. ' ' .. x .. '\n'); spikeN = spikeN + 1 end
  , timeStep = timeStep
  , subSteps = 1
  , n = 3.5
  , model = numint.euler
  , params = {pre_tau = 15, post_tau = 15, a_minus = 0.1, a_plus = 0.1}
  , inputType = 'spikes'
  }

--[[remove latent-latent connections
local latentStart = 950*networkScale
local latentEnd = 1000*networkScale
sr._connections:sub(latentStart, latentEnd, latentStart, latentEnd):fill(0)
sr._weights:sub(latentStart, latentEnd, latentStart, latentEnd):fill(0)]]

local potentials = torch.zeros(totalSteps, sr._nTotal)
local spikes1 = torch.zeros(window, outputs)
local spikes2 = torch.zeros(window, outputs)
local spikes3 = torch.zeros(window, outputs)
local monitoreds = {inputs, inputs + 1}
local monitorVars = {'_potential', '_gAMPA', '_gNMDA', '_gGABAA', '_gGABAB', '_recovery', '_preTrace'}
local monitors = {}

for _, m in pairs(monitoreds) do
  monitors[m] = {}
  for _, v in pairs(monitorVars) do
    monitors[m][v] = torch.zeros(totalSteps)
  end
end

local function updMonitors(i)
  for _, m in pairs(monitoreds) do
    for _, v in pairs(monitorVars) do
      monitors[m][v][i] = sr[v][m]
    end
  end
end

--gnuplot.imagesc(sr._weights ,'color')

print('running simulation')
local time1 = os.time()

for i = 1, exciteSteps, 1 do
  sr:forward(torch.cmul(torch.lt(torch.rand(inputs), infreq * timeStep):double() * inspike, torch.rand(inputs) / 5 + 0.8))
  potentials[i] = sr._potential

  spikes1[(i % window) + 1] = sr._spikes:index(1, sr._outputs)

  updMonitors(i)
  if i % printEvery == 0 then print(i .. '/' .. totalSteps .. ', ' .. spikeN .. ' spikes, ' .. torch.sum(spikes1) / (timeStep * window * outputs) .. ' average output spiking frequency (Hz)') end
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
print('ran ' .. (totalSteps * sr._timeStep) / 1000 .. ' seconds of simulation in ' .. os.time() - time1 .. ' seconds')

file:flush()
file:close()

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
--[[
for k, m in pairs(monitoreds) do
  for _, v in pairs(monitorVars) do
    gnuplot.figure()
    gnuplot.plot(v .. '_' .. k, torch.range(0, totalSteps - 1) * timeStep, monitors[m][v], '-')
  end
end
]]
gnuplot.figure()
gnuplot.raw("plot 'data.dat' with dots")
gnuplot.figure()
gnuplot.imagesc(potentials:t() ,'color')
gnuplot.figure()
gnuplot.imagesc(sr._weights ,'color')
gnuplot.figure()
gnuplot.imagesc(sr._x)

