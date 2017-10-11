--[[
Produce results of Dynamical systems in neuroscience examples of 6 typical neuron types
]]

require 'spikingreservoir_rk'
require 'gnuplot'
local models = require 'models'
local numint = require 'numint'

local models =
  { ['Regular Spiking'] = {
      -- Figure 8.12 on page 283
      model = models.regularSpiking(1, {
        C = models.constant(100)
      , k = models.constant(0.7)
      , v_r = models.constant(-60)
      , v_t = models.constant(-40)
      , v_peak = models.constant(35)
      , a = models.constant(0.03)
      , b = models.constant(-2)
      , c = models.constant(-50)
      , d = models.constant(100)
      })
    , currents = { 60, 70, 85, 100 }
    , time = 600
    , input = function(totalSteps)
                return torch.cat({ torch.zeros(totalSteps*0.1, 1)
                                 , torch.ones(totalSteps*0.85, 1)
                                 , torch.zeros(totalSteps*0.05, 1)}, 1)
              end
    }
  , ['Intrinsically Bursting'] = {
      -- 8.19 on page 290
      model = models.regularSpiking(1, {
        C = models.constant(150)
      , k = models.constant(1.2)
      , v_r = models.constant(-75)
      , v_t = models.constant(-45)
      , v_peak = models.constant(50)
      , a = models.constant(0.01)
      , b = models.constant(5)
      , c = models.constant(-56)
      , d = models.constant(130)
      })
    , currents = { 300, 370, 500, 550 }
    , time = 600
    , input = function(totalSteps)
                return torch.cat({ torch.zeros(totalSteps*0.1, 1)
                                 , torch.ones(totalSteps*0.90, 1)}, 1)
              end
    }
  , ['Chattering Neuron'] = {
      -- Figure 8.23 on page 295
      model = models.regularSpiking(1, {
        C = models.constant(50)
      , k = models.constant(1.5)
      , v_r = models.constant(-60)
      , v_t = models.constant(-40)
      , v_peak = models.constant(25)
      , a = models.constant(0.03)
      , b = models.constant(1)
      , c = models.constant(-40)
      , d = models.constant(150)
      })
    , currents = { 200, 300, 400, 600 }
    , time = 220
    , input = function(totalSteps)
                return torch.cat({ torch.zeros(totalSteps*0.05, 1)
                                 , torch.ones(totalSteps*0.90, 1)
                                 , torch.zeros(totalSteps*0.05, 1)}, 1)
              end
    }
  , ['Low-Threshold Spiking'] = {
      -- Figure 8.23 on page 295
      model = models.regularSpiking(1, {
        C = models.constant(100)
      , k = models.constant(1)
      , v_r = models.constant(-56)
      , v_t = models.constant(-42)
      , v_peak = models.constant(40) -- Should have the -0.1u term,
                                     -- but my framework does not easily allow it
      , a = models.constant(0.03)
      , b = models.constant(8)
      , c = models.constant(-53) -- Similarily should have +0.04u
      , d = models.constant(20) -- Should be capped at 670
      })
    , currents = { 100, 125, 200, 300 }
    , time = 300
    , input = function(totalSteps)
                return torch.cat({ torch.zeros(totalSteps*0.05, 1)
                                 , torch.ones(totalSteps*0.95, 1)}, 1)
              end
    }
  , ['Fast Spiking'] = {
      -- Figure 8.27 on page 299
      -- Would require a whole different model to reproduce accurately
      model = models.regularSpiking(1, {
        C = models.constant(20)
      , k = models.constant(1)
      , v_r = models.constant(-55)
      , v_t = models.constant(-40)
      , v_peak = models.constant(25)
      , a = models.constant(0.020)
      , b = models.constant(0.025)
      , c = models.constant(-45)
      , d = models.constant(0)
      })
    , currents = { 73.2, 100, 200, 400 }
    , time = 120
    , input = function(totalSteps)
                return torch.cat({ torch.zeros(totalSteps*0.05, 1)
                                 , torch.ones(totalSteps*0.95, 1)}, 1)
              end
    }
  , ['Late Spiking'] = {
      -- Figure 8.28 on page 300
      -- Would require dendritic current
      model = models.regularSpiking(1, {
        C = models.constant(20)
      , k = models.constant(0.3)
      , v_r = models.constant(-66)
      , v_t = models.constant(-40)
      , v_peak = models.constant(30)
      , a = models.constant(0.17)
      , b = models.constant(5)
      , c = models.constant(-45)
      , d = models.constant(100)
      })
    , currents = { 125, 150, 200 }
    , time = 800
    , input = function(totalSteps)
                return torch.cat({ torch.zeros(totalSteps*0.125, 1)
                                 , torch.ones(totalSteps*0.75, 1)
                                 , torch.zeros(totalSteps*0.125, 1)}, 1)
              end
    }
  }

for modelName, modelParams in pairs(models) do
for _, inCurrent in pairs(modelParams.currents) do

local timeScale = 1
local spikeN = 0
local timeStep = 1e-4*timeScale
local totalTime = modelParams.time--600 -- ms
local totalSteps = totalTime/(1000*timeStep/timeScale)/timeScale--math.floor(testLength / 2 + 0.5)
local inputs = 1
local outputs = 1
--local inCurrent = 60
local printEvery = 100

local sr = nn.SpikingReservoir
  { neurons = {modelParams.model}
  , connectivity = 0.2
  , inputs = torch.range(1, 1):long()
  , outputs = torch.range(1, 1):long()
  , spikeCallback = function(x, time) spikeN = spikeN + 1 end
  , timeStep = timeStep
  , subSteps = 1
  , n = 1
  , method = numint.RK4
  }

local potentials = torch.zeros(totalSteps, sr._nTotal)
local input = modelParams.input(totalSteps) * inCurrent--torch.cat({torch.zeros(totalSteps*0.1, inputs), torch.ones(totalSteps*0.85, inputs)*inCurrent, torch.zeros(totalSteps*0.05, inputs)}, 1)
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

print('running simulation for ' .. modelName)
local time1 = os.time()

for i = 1, totalSteps do
  sr:forward(input[i])
  potentials[i] = sr._potential

  updMonitors(i)
  if i % printEvery == 0 then print(i .. '/' .. totalSteps .. ', ' .. spikeN .. ' spikes') end
end

print('simulation ended')
print('ran ' .. (totalSteps) * sr._timeStep .. ' seconds of simulation in ' .. os.time() - time1 .. ' seconds')

print('total ' .. spikeN .. ' spikes')

gnuplot.figure()
gnuplot.raw('set multiplot layout 2,1')
gnuplot.raw('set title "' .. modelName .. '"')
gnuplot.plot('I = ' .. inCurrent, torch.range(0, totalSteps - 1) * timeStep, monitors._potential, '-')
gnuplot.plot('phase', monitors._potential, monitors._recovery, '-')

end
end
