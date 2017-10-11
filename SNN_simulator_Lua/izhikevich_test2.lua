--Disregard this
-- The goal of this test is to reproduce the results in 'figure1.m'
-- i.e. the 20 basic neurocomputational properties
-- 0.04v² + 5*v + 140 - u + I
-- (k(v - v_t)(v - v_r) - u + I) / C
--> (k(v² - v*v_r - v*v_t + v_t*v_r) - u + I) / C
--[[
k = 0.04
v_r + v_t = -125
v_t*v_r = 3500
-83 and -42
a*(b*v - u)
a*(b(v - v_r) - u)
]]

package.path = package.path .. ';?.lua'

require 'spikingreservoir_rk'
require 'gnuplot'
local models = require 'models'
local numint = require 'numint'

local models =
  { ['(A) Tonic Spiking'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.02)
      , b = models.constant(0.2)
      , c = models.constant(-65)
      , d = models.constant(6)
      })
    , v_start = -70
    , I = 14
    , time = 100
    }
  , ['(B) Phasic Spiking'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.02)
      , b = models.constant(0.25)
      , c = models.constant(-65)
      , d = models.constant(6)
      })
    , v_start = -64
    , I = 0.5
    , time = 200
    }
  , ['(C) Tonic Bursting'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.02)
      , b = models.constant(0.2)
      , c = models.constant(-50)
      , d = models.constant(2)
      })
    , v_start = -70
    , I = 15
    , time = 220
    }
  , ['(D) Phasic Bursting'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.02)
      , b = models.constant(0.25)
      , c = models.constant(-55)
      , d = models.constant(0.05)
      })
    , v_start = -64
    , I = 0.6
    , time = 200
    }
  , ['(E) Mixed Mode'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.02)
      , b = models.constant(0.2)
      , c = models.constant(-55)
      , d = models.constant(4)
      })
    , v_start = -70
    , I = 10
    , time = 160
    }
  , ['(F) Spike Frequency Adaptation'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.01)
      , b = models.constant(0.2)
      , c = models.constant(-65)
      , d = models.constant(8)
      })
    , v_start = -70
    , I = 30
    , time = 85
    }
  , ['(G) Class 1 Excitable'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.02)
      , b = models.constant(-0.1)
      , c = models.constant(-55)
      , d = models.constant(6)
      })
    , v_start = -60
    , I = function(steps)
        return torch.cat( torch.zeros(0.1*steps, 1)
                        , torch.linspace(0, 270, 0.9*steps)*0.075, 1)
      end
    , time = 300
    }
  , ['(H) Class 2 Excitable'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.2)
      , b = models.constant(0.26)
      , c = models.constant(-65)
      , d = models.constant(0)
      })
    , v_start = -64
    , I = function(steps)
        return -0.5 + torch.cat( torch.zeros(0.1*steps, 1)
                               , torch.linspace(0, 270, 0.9*steps)*0.015, 1)
      end
    , time = 300
    }
  , ['(I) Spike Latency'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.02)
      , b = models.constant(0.2)
      , c = models.constant(-65)
      , d = models.constant(6)
      })
    , v_start = -70
    , I = function(steps)
        return torch.cat( torch.zeros(0.1*steps, 1)
                        , torch.cat( 7.04*torch.ones(0.03*steps, 1)
                                   , torch.zeros(0.87*steps, 1), 1), 1)
      end
    , time = 100
    }
  , ['(J) Subthreshold Oscillations'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.05)
      , b = models.constant(0.26)
      , c = models.constant(-60)
      , d = models.constant(0)
      })
    , v_start = -62
    , I = function(steps)
        return torch.cat( torch.zeros(0.1*steps, 1)
                        , torch.cat( 2*torch.ones(0.025*steps, 1)
                                   , torch.zeros(0.875*steps, 1), 1), 1)
      end
    , time = 200
    }
  , ['(K) Resonator'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.1)
      , b = models.constant(0.26)
      , c = models.constant(-60)
      , d = models.constant(-1)
      })
    , v_start = -62
    , I = function(steps)
        return 0.65 * torch.cat({
            torch.zeros(0.1*steps, 1)
          , torch.ones(0.01*steps, 1)
          , torch.zeros(0.05*steps, 1)
          , torch.ones(0.01*steps, 1)
          , torch.zeros(0.53*steps, 1)
          , torch.ones(0.01*steps, 1)
          , torch.zeros(0.1*steps, 1)
          , torch.ones(0.01*steps, 1)
          , torch.zeros(0.18*steps, 1)
          }, 1)
      end
    , time = 400
    }
  , ['(L) Integrator'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.02)
      , b = models.constant(-0.1)
      , c = models.constant(-55)
      , d = models.constant(6)
      })
    , v_start = -60
    , I = function(steps)
        return 9 * torch.cat({
            torch.zeros(0.1*steps, 1)
          , torch.ones(0.02*steps, 1)
          , torch.zeros(0.03*steps, 1)
          , torch.ones(0.02*steps, 1)
          , torch.zeros(0.53*steps, 1)
          , torch.ones(0.02*steps, 1)
          , torch.zeros(0.1*steps, 1)
          , torch.ones(0.02*steps, 1)
          , torch.zeros(0.16*steps, 1)
          }, 1)
      end
    , time = 100
    }
  , ['(M) Rebound Spike'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.03)
      , b = models.constant(0.25)
      , c = models.constant(-60)
      , d = models.constant(4)
      })
    , v_start = -64
    , I = function(steps)
        return -15 * torch.cat({
            torch.zeros(0.1*steps, 1)
          , torch.ones(0.025*steps, 1)
          , torch.zeros(0.875*steps, 1)
          }, 1)
      end
    , time = 200
    }
  , ['(N) Rebound Burst'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.03)
      , b = models.constant(0.25)
      , c = models.constant(-52)
      , d = models.constant(0)
      })
    , v_start = -64
    , I = function(steps)
        return -15 * torch.cat({
            torch.zeros(0.1*steps, 1)
          , torch.ones(0.025*steps, 1)
          , torch.zeros(0.875*steps, 1)
          }, 1)
      end
    , time = 200
    }
  , ['(O) Threshold Variability'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.03)
      , b = models.constant(0.25)
      , c = models.constant(-60)
      , d = models.constant(4)
      })
    , v_start = -64
    , I = function(steps)
        return torch.cat({
            torch.zeros(0.1*steps, 1)
          , torch.ones(0.05*steps, 1)
          , torch.zeros(0.55*steps, 1)
          , torch.ones(0.05*steps, 1) * -6
          , torch.zeros(0.05*steps, 1)
          , torch.ones(0.05*steps, 1)
          , torch.zeros(0.15*steps, 1)
          }, 1)
      end
    , time = 100
    }
  , ['(P) Bistability'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.1)
      , b = models.constant(0.26)
      , c = models.constant(-60)
      , d = models.constant(0)
      })
    , v_start = -61
    , I = function(steps)
        local spikeLength = 0.01
        return 0.24 + torch.cat({
            torch.zeros(0.125*steps, 1)
          , torch.ones(spikeLength*steps, 1)
          , torch.zeros(0.585*steps, 1)
          , torch.ones(spikeLength*steps, 1)
          , torch.zeros(0.27*steps, 1)
          }, 1)
      end
    , time = 300
    }
  , ['(Q) Depolarizing After-Potential'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(1)
      , b = models.constant(0.2)
      , c = models.constant(-60)
      , d = models.constant(-21)
      })
    , v_start = -70
    , I = function(steps)
        return 0.24 + torch.cat({
            torch.zeros(0.20*steps, 1)
          , torch.ones(0.02*steps, 1) * 20
          , torch.zeros(0.78*steps, 1)
          }, 1)
      end
    , time = 50
    }
  , ['(R) Accommodation'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(0.02)
      , b = models.constant(1)
      , c = models.constant(-55)
      , d = models.constant(4)
      })
    , v_start = -65
    , I = function(steps)
        local ramp1 = torch.linspace(0, 8, 0.5*steps)
        local ramp2 = torch.linspace(0, 4, 0.03*steps)
        return torch.cat({
            ramp1:reshape(ramp1:nElement(), 1)
          , torch.zeros(0.25*steps, 1)
          , ramp2:reshape(ramp2:nElement(), 1)
          , torch.zeros(0.27*steps, 1)
          }, 1)
      end
    , time = 400
    }
  , ['(S) Inhibition-Induced Spiking'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(-0.02)
      , b = models.constant(-1)
      , c = models.constant(-60)
      , d = models.constant(8)
      })
    , v_start = -63.8
    , I = function(steps)
        return 75 + torch.cat({
            torch.ones(0.125*steps, 1) * 5
          , torch.zeros(0.5*steps, 1)
          , torch.ones(0.375*steps, 1) * 5
          }, 1)
      end
    , time = 400
    }
  , ['(T) Inhibition-Induced Bursting'] = {
      model = models.regularSpiking(1, {
        v_peak = models.constant(30)
      , a = models.constant(-0.026)
      , b = models.constant(-1)
      , c = models.constant(-45)
      , d = models.constant(-2)
      })
    , v_start = -63.8
    , I = function(steps)
        return 75 + torch.cat({
            torch.ones(0.125*steps, 1) * 5
          , torch.zeros(0.5*steps, 1)
          , torch.ones(0.375*steps, 1) * 5
          }, 1)
      end
    , time = 400
    }
  }

local selection = {}
selection['(Q) Depolarizing After-Potential'] = models['(Q) Depolarizing After-Potential']

--models = selection

for modelName, modelParams in pairs(models) do

local timeScale = 5
local spikeN = 0
local timeStep = 1e-6*timeScale
local totalTime = modelParams.time -- ms
local totalSteps = totalTime/(1000*timeStep/timeScale)/timeScale
local inputs = 1
local outputs = 1

local input
if type(modelParams.I) == 'number' then
  input = torch.cat( torch.zeros(totalSteps*0.1, inputs)
                   , torch.ones(totalSteps*0.9, inputs)*modelParams.I, 1)
else
  input = modelParams.I(totalSteps)
end

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
  , startingPotential = modelParams.v_start
  }

if modelName == '(G) Class 1 Excitable' or
   modelName == '(L) Integrator' then
  sr._Dv = function(t, y)
    return 0.04 * torch.pow(y, 2) + 4.1 * y + 108 - sr._recovery + sr._I-- + input
  end
else
  sr._Dv = function(t, y)
--print(input)
    return 0.04 * torch.pow(y, 2) + 5 * y + 140 - sr._recovery + sr._I-- + input
  end
end

sr._Du = function(t, y)
  return torch.cmul(sr._a, (torch.cmul(sr._b, sr._potential) - y))
end

if modelName == '(R) Accommodation' then
  sr._recovery:fill(-16)
else
  sr._recovery = torch.cmul(sr._potential, sr._b)
end

local potentials = torch.zeros(totalSteps, sr._nTotal)
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
  if i % printEvery == 0 then
    print(i .. '/' .. totalSteps .. ', ' .. spikeN .. ' spikes')
    --print(torch.norm(sr._potential))
  end
end

print('simulation ended')
print('ran ' .. (totalSteps) * sr._timeStep / 1000 .. ' seconds of simulation in ' .. os.time() - time1 .. ' seconds')

print('total ' .. spikeN .. ' spikes')

gnuplot.figure()
gnuplot.raw('set multiplot layout 2,1')
gnuplot.raw('set title "' .. modelName .. '"')
gnuplot.plot('potential', torch.range(0, totalSteps - 1) * timeStep, monitors._potential, '-')
gnuplot.raw('set title ""')
gnuplot.plot('phase', monitors._potential, monitors._recovery, '-')

end
