local models = {}

function models.constant(x)
  return function() return x end
end 

models.modify = function(default, modifications)
  modifications = modifications or {}
  local result = {}

  for k, v in pairs(default) do
    result[k] = v
  end
  for k, v in pairs(modifications) do
    result[k] = v
  end
  return result
end

--[[
LS (Latent spiking)
RS (Regular spiking)
FS (Fast spiking)
LTS (Low-threshold spiking)
TC (Thalamocortical)
TI (Thalamic interneurons)
RTN (Reticular thalamic nucleus)
]]
--[[
proportions or neurons
p2/3 26%
p6(L4) 13.6
ss4(L4) 9.2%
ss4(L2/3) 9.2%
p4 9.2%

latentSpiking
nb1	1.5
nb2/3	4.2
nb4	1.5
nb5	0.8
nb6	2
total	10
]]

--[[
a is the recovery time
The sign of b determines whether u is an amplifying or a resonant variable
]]

function models.latentSpiking(n, config)
  -- nb1
  local model =
    { C = models.constant(20)
    , k = models.constant(0.3)
    , v_r = models.constant(-66)
    , v_t = models.constant(-40)
    , v_peak = models.constant(30)
    , a = models.constant(0.17)
    , b = models.constant(5)
    , c = models.constant(-45)
    , d = models.constant(100)
    , tau_x = models.constant(100)
    , p = models.constant(1.5)
    , g_AMPA = models.constant(1)
    , g_NMDA = models.constant(1)
    , g_GABAA = models.constant(0)
    , g_GABAB = models.constant(0)
    }
  return {n = n, model = models.modify(model, config)}
end

function models.regularSpiking(n, config)
  -- p23
  local model =
    { C = models.constant(100)
    , k = models.constant(3)
    , v_r = models.constant(-60)
    , v_t = models.constant(-50)
    , v_peak = models.constant(50)
    , a = models.constant(0.01)
    , b = models.constant(5)
    , c = models.constant(-60)
    , d = models.constant(400)
    , tau_x = models.constant(100)
    , p = models.constant(0.6)
    , g_AMPA = models.constant(1)
    , g_NMDA = models.constant(1)
    , g_GABAA = models.constant(0)
    , g_GABAB = models.constant(0)
    }
  return {n = n, model = models.modify(model, config)}
end

-- Inhibitory

function models.fastSpiking(n, config)
  local model =
    { C = models.constant(20)
    , k = models.constant(1)
    , v_r = models.constant(-55)
    , v_t = models.constant(-40)
    , v_peak = models.constant(25)
    , a = models.constant(0.15)
    , b = models.constant(8)
    , c = models.constant(-55)
    , d = models.constant(200)
    , tau_x = models.constant(150)
    , p = models.constant(0.6)
    , g_AMPA = models.constant(0)
    , g_NMDA = models.constant(0)
    , g_GABAA = models.constant(1)
    , g_GABAB = models.constant(1)
    }
  return {n = n, model = models.modify(model, config)}
end

function models.lowTreshold(n, config)
  local model =
    { C = models.constant(100)
    , k = models.constant(1)
    , v_r = models.constant(-56)
    , v_t = models.constant(-42)
    , v_peak = models.constant(40)
    , a = models.constant(0.03)
    , b = models.constant(8)
    , c = models.constant(-50)
    , d = models.constant(20)
    , tau_x = models.constant(100)
    , p = models.constant(1.5)
    , g_AMPA = models.constant(0)
    , g_NMDA = models.constant(0)
    , g_GABAA = models.constant(1)
    , g_GABAB = models.constant(1)
    }
  return {n = n, model = models.modify(model, config)}
end

return models
