require 'nn'
require 'cutorch'
local numint = require 'numint'
local models = require 'models'

local SpikingReservoir, parent = torch.class('nn.SpikingReservoir','nn.Module')

function SpikingReservoir:__init(config)
  config = config or {}
  assert(torch.type(config) == 'table' and not config[1], 
    "Constructor requires key-value arguments")

  local defaultParams =
    { a_minus = 1
    , a_plus = 1
    , A_minus = function(x) return x end
    , A_plus = function(x) return (1 - x) end
    , pre_tau = 5
    , post_tau = 5
    }

  local args, neurons, connectivity, inputs, outputs, spikeCallback, timeStep, subSteps, params, n, method, startingPotential, inputType = xlua.unpack(
      {config},
      'SpikingReservoir', 
      'A reservoir of spiking neurons',
      {arg='neurons', type='table',
       help='Table of numbers and types of neurons'},
      {arg='connectivity', type='number',
       help='Connectivity parameter, e.g. the probability that any given neurons are connected'},
      {arg='inputs', type='torch.LongTensor',
       help='Indices of input neurons.'},
      {arg='outputs', type='torch.LongTensor',
       help='Indices of output neurons.'},
      {arg='spikeCallback', type='function', req=false,
       help='Function to call on spikes'},
      {arg='timeStep', type='number',
       help='Length (in time) of a simulation step'},
      {arg='subSteps', type='number',
       help='Number of steps in a simgle simulation step'},
      {arg='params', type='table', default={},
       help='Parameters for STDP'},
      {arg='n', type='number', default=1,
       help='Number of synapses between neurons. Simulated as simply a factor on synaptic currents.'},
      {arg='method', type='function', default=numint.euler,
       help='Numerical integration method'},
      {arg='startingPotential', type='number', default=-65,
       help='Initial value for membrane potential'},
      {arg='inputType', type='string', default='current',
       help='Input type. "current" for continous current, "spikes" for AMPA+NMDA spikes.'}
  )

  params = models.modify(defaultParams, params)
  -- These are the scales of the parameters in the system
  -- These are a recent addition; feel free to try resetting them to 1
  --[[ (i.e. disabling them)
  self._faradScale = 1e-12  -- picofarads
  self._voltScale = 1e-3    -- millivolts
  self._ampereScale = 1e-12 -- picoamperes
  self._timeScale = 1e-3    -- milliseconds
  local kilo = 1e3          -- used to turn kHz to Hz
  --]]
  ---[[
  self._faradScale = 1  -- picofarads
  self._voltScale = 1   -- millivolts
  self._ampereScale = 1 -- picoamperes
  self._timeScale = 1   -- milliseconds
  local kilo = 1        -- used to turn kHz to Hz
  --]]

  local nTotal = 0
  for _, v in pairs(neurons) do
    nTotal = nTotal + v.n
  end
  assert(nTotal > 0, 'SpikingReservoir.__init: There must be at least one neuron!')
  self._nTotal = nTotal

  self._C = torch.zeros(nTotal) 
  self._k = torch.zeros(nTotal)
  self._v_r = torch.zeros(nTotal)
  self._v_t = torch.zeros(nTotal)
  self._v_peak = torch.zeros(nTotal)
  self._a = torch.zeros(nTotal)
  self._b = torch.zeros(nTotal)
  self._c = torch.zeros(nTotal)
  self._d = torch.zeros(nTotal)
  self._tau_x = torch.zeros(nTotal, nTotal)
  self._p = torch.zeros(nTotal)
  self._gAMPA_recep = torch.zeros(nTotal)
  self._gNMDA_recep = torch.zeros(nTotal)
  self._gGABAA_recep = torch.zeros(nTotal)
  self._gGABAB_recep = torch.zeros(nTotal)

  local i = 0
  for _, v in pairs(neurons) do
    for j = i+1, i+v.n, 1 do
      self._C[j] = v.model.C() * self._faradScale
      self._k[j] = v.model.k() * self._ampereScale / self._voltScale / self._voltScale
      self._v_r[j] = v.model.v_r() * self._voltScale
      self._v_t[j] = v.model.v_t() * self._voltScale
      self._v_peak[j] = v.model.v_peak() * self._voltScale
      self._a[j] = v.model.a() / self._timeScale * kilo
      self._b[j] = v.model.b() * self._ampereScale / self._voltScale
      self._c[j] = v.model.c() * self._voltScale
      self._d[j] = v.model.d() * self._ampereScale
      self._tau_x[{{},j}] = v.model.tau_x() * self._timeScale
      self._p[j] = 1 - v.model.p()
      self._gAMPA_recep[j] = v.model.g_AMPA()
      self._gNMDA_recep[j] = v.model.g_NMDA()
      self._gGABAA_recep[j] = v.model.g_GABAA()
      self._gGABAB_recep[j] = v.model.g_GABAB()
    end

    i = i + v.n
  end

  self._spikeCallback = spikeCallback
  self._timeStep = timeStep*1000
  self._subSteps = subSteps
  self._preTrace = torch.zeros(nTotal)
  self._postTrace = torch.zeros(nTotal)
  self._potential = torch.zeros(nTotal):fill(startingPotential) * self._voltScale
  self._I = torch.zeros(nTotal)
  self._recovery = torch.zeros(nTotal)
  self._x = torch.ones(nTotal, nTotal)
  self._gAMPA = torch.zeros(nTotal)
  self._gNMDA = torch.zeros(nTotal)
  self._gGABAA = torch.zeros(nTotal)
  self._gGABAB = torch.zeros(nTotal)
  self._spikes = torch.zeros(nTotal)
  self._zeroes = torch.zeros(nTotal, nTotal)
  
  -- Construct connection and weight matrices
  self._connections = torch.lt( torch.rand(nTotal, nTotal)
                              , torch.ones(nTotal, nTotal) * connectivity)
  -- Zero the connections to self (diagonal)
  self._connections = self._connections - torch.diag(torch.diag(self._connections))
  self._connections_d = self._connections:double()
  
  self._weights = 0.5 + 0.5 * torch.rand(nTotal, nTotal)
  self._weights:cmul(self._connections:double())
  --[[local nonzeroes = self._connections:nonzero()
  local entriesc = {}
  local entriesw = {}
  for i = 1, nonzeroes:size(1) do
    local k, l = nonzeroes[i][1], nonzeroes[i][2]
    entriesc[i] = { k, l, 1 }
    entriesw[i] = { k, l, self._weights[k][l] }
  end
  self._connections = torch.DoubleTensor(entriesc)
  self._weights = torch.DoubleTensor(entriesw)
  print(self._connections)]]

  self._inputs = inputs
  self._outputs = outputs
  self._params = params
  self._n = n
  self._time = 0
  self._method = method -- Numeric integration method

  -- Reverse potentials and other constants
  self._E_NMDA = 80 * self._voltScale
  self._NMDA_divisor = 60 * self._voltScale
  self._E_GABAA = 70 * self._voltScale
  self._E_GABAB = 90 * self._voltScale

  -- Time constants
  self._gAMPA_tau = 5 * self._timeScale
  self._gNMDA_tau = 150 * self._timeScale
  self._gGABAA_tau = 6 * self._timeScale
  self._gGABAB_tau = 150 * self._timeScale
  self._pre_tau = params.pre_tau * self._timeScale
  self._post_tau = params.post_tau * self._timeScale

  -- Triplet STDP
  self._weight_e -- Bitmask of excitatory weights
  self._weight_i -- Bitmask of inhibitory weights
  self._A = -- A constant parameter
  self._beta
  self._delta
  -- Synaptic traces
  self._zp = torch.zeros(nTotal)
  self._zp_tau = 
  self._zm = torch.zeros(nTotal)
  self._zm_tau = 
  self._zslow = torch.zeros(nTotal)
  self._zslow_tau = 
  -- Consolidation dynamics
  self._ref_weight = self._weights
  self._P
  self._wP = 0.5
  self._cons_tau = 20*60*1000

  -- Create closures of derivates
  self._Dv = self:Dv() -- Derivate of membrane potential
  self._Du = self:Du() -- Derivate of recovery variable
  self._Dx = self:Dx() -- Derivate of conductance factor
  self._Dg = self:Dg() -- Derivate of neurotransmitter receptors

  if inputType == 'current' then
    self._inputHandler = function(input) self._I:indexCopy(1, self._inputs, input) end
  elseif inputType == 'spikes' then
    self._inputHandler = function(input)
      self._gAMPA:indexAdd(1, self._inputs, input)
      self._gNMDA:indexAdd(1, self._inputs, input)
    end
  elseif inputType == 'freqs' then
    self._inputHandler = function(input)
      self._gAMPA:indexAdd(1, self._inputs,
        torch.lt(torch.rand(self._inputs:nElement()), input):double())
      self._gNMDA:indexAdd(1, self._inputs,
        torch.lt(torch.rand(self._inputs:nElement()), input):double())
    end
  else
    error('Unknown input type')
  end

  self._outputHandler = function(output)
      self._gAMPA:indexAdd(1, self._outputs, output)
      self._gNMDA:indexAdd(1, self._outputs, output)      
    end
end

-- Derivate of potential
function SpikingReservoir:Dv()
  return function(t, y, input)
    local tmp = torch.pow((y + self._E_NMDA) / self._NMDA_divisor, 2)
    local vmv_r = y - self._v_r

    -- This is the synaptic current
    local I = torch.cmul(
                y
              , self._gAMPA + torch.cmul(self._gNMDA, torch.cdiv(tmp, 1 + tmp))
              ) -- cmul
              + torch.cmul(self._gGABAA, y + self._E_GABAA)
              + torch.cmul(self._gGABAB, y + self._E_GABAB)
              + self._I

--    I:indexAdd(1, self._inputs, -input)

    -- Dendritic current is not simulated

    -- Here, the current I is added to the potential in equations in sources, but I is
    -- the sum of negations of the dendritic and synaptic currents.
    -- For performance reasons we do not want to calculate negations in the previous formula.
      return torch.cdiv(
      torch.cmul(
        self._k
      , torch.cmul(
          vmv_r
        , y - self._v_t
        ) -- cmul
      ) -- cmul
    - self._recovery
    - I * self._ampereScale
    , self._C
    ) -- cdiv
  end
end

function SpikingReservoir:Du()
  return function(t, y, vmv_r)
    return torch.cmul(self._a
         , torch.cmul(self._b, vmv_r) - y)
  end
end

function SpikingReservoir:Dx()
  return function(t, y)
      return torch.cdiv(1 - y, self._tau_x)
  end
end

function SpikingReservoir:Dg()
  return function(t, y, tau)
    return -y / tau
  end
end

-- Calculate a single timestep
function SpikingReservoir:timeStep()
  self._spikes = torch.gt(self._potential, self._v_peak)
  local spikes_d = self._spikes--:double() -- Store type conversion for later use

  if not self._cuda then
    spikes_d = spikes_d:double()
  end

  self._outSpikes:add(spikes_d:index(1, self._outputs))

  -- Reset spiked neurons
  self._potential:maskedCopy(self._spikes, self._c[self._spikes])
  self._recovery:add(torch.cmul(spikes_d, self._d))

  -- Update potentials
  
  self._potential =
    self._method(self._Dv, self._time, self._potential, self._timeStep)

  -- Update recovery
  local vmv_r = self._potential - self._v_r -- Constant in the next calculation
  self._recovery =
    self._method(self._Du, self._time, self._recovery, self._timeStep, vmv_r)

  -- Add spikes to conductance factor
  local p_spike_t = torch.repeatTensor(torch.cmul(spikes_d, self._p), self._nTotal, 1):t()

  --if self._cuda then
  --  p_spike_t = p_spike_t:cuda()
  --end

  self._x:cmul(1 - torch.cmul(self._connections_d, p_spike_t))

  -- Update conductance factor
  self._x = self._method(self._Dx, self._time, self._x, self._timeStep)

  -- Add spikes to receptors
  local spikes_d_t = spikes_d:reshape(1,self._nTotal) * self._n
  --local spikes_d_x = torch.cmul(spikes_d, self._x)
  self._gAMPA:add(
    torch.mm(torch.cmul(spikes_d_t, self._gAMPA_recep), torch.cmul(self._weights, self._x)))
  self._gNMDA:add(
    torch.mm(torch.cmul(spikes_d_t, self._gNMDA_recep), torch.cmul(self._weights, self._x)))
  self._gGABAA:add(
    torch.mm(torch.cmul(spikes_d_t, self._gGABAA_recep), torch.cmul(self._weights, self._x)))
  self._gGABAB:add(
    torch.mm(torch.cmul(spikes_d_t, self._gGABAB_recep), torch.cmul(self._weights, self._x)))

  -- Update receptors
  self._gAMPA =
    self._method(self._Dg, self._time, self._gAMPA, self._timeStep, self._gAMPA_tau)
  self._gNMDA =
    self._method(self._Dg, self._time, self._gNMDA, self._timeStep, self._gNMDA_tau)
  self._gGABAA =
    self._method(self._Dg, self._time, self._gGABAA, self._timeStep, self._gGABAA_tau)
  self._gGABAB =
    self._method(self._Dg, self._time, self._gGABAB, self._timeStep, self._gGABAB_tau)
  
  -- Add spikes to weights
  local function nonzero(tensor)
    local tmp = torch.nonzero(tensor)
    -- Result is n x 1 tensor: reshape it to a length n vector
    -- squeeze() doesn't work as a scalar tensor (n=1) gets squeezed to a number
    return tmp:reshape(tmp:nElement())
  end

  --local indices = nonzero(self._spikes)
  --if indices:nElement() > 0 then
    --local zeroes = torch.zeros(self._nTotal, self._nTotal)
    --if self._cuda then
    --  zeroes = zeroes:cuda()
    --end

    -- Incrase weights for presynaptic spikes
    -- A_plus(x) -> 1 - x
--[[    self._weights:add(
        self._weights:clone():apply(self._params.A_plus):cmul(
        torch.cmul(self._connections_d
      , torch.addr(self._zeroes
        , self._preTrace, spikes_d))))]]
    self._weights:add(
        self._weights:clone():neg():add(1):cmul(
        torch.cmul(self._connections_d
      , torch.addr(self._zeroes
        , self._preTrace, spikes_d))))
--          torch.repeatTensor(torch.cmul(spikes_d, self._preTrace), self._nTotal, 1):t()
--        , torch.repeatTensor(spikes_d, self._nTotal, 1)))))

    -- Decrease weights for postsynaptic spikes
    -- A_minus(x) -> x
--[[    self._weights = self._weights
      - self._weights:clone():apply(self._params.A_minus):cmul(
        torch.cmul(self._connections_d
      , torch.addr(self._zeroes, spikes_d, self._postTrace)))]]
    self._weights:csub(
        self._weights:clone():cmul(
        torch.cmul(self._connections_d
      , torch.addr(self._zeroes, spikes_d, self._postTrace))))
    --torch.repeatTensor(torch.cmul(spikes_d, self._postTrace), self._nTotal, 1)))

    -- Clamp weights to [0,1] range
    self._weights:clamp(0, 1)
  --end

  -- Update pre/postsynaptic traces and
  -- add spikes to pre/postsynaptic traces
  -- Perhaps we need only one trace if a+ = a-
  self._preTrace:add(self._params.a_plus, spikes_d)
  self._postTrace:add(self._params.a_minus, spikes_d)
  self._preTrace =
    self._method(self._Dg, self._time, self._preTrace, self._timeStep, self._pre_tau)
  self._postTrace =
    self._method(self._Dg, self._time, self._postTrace, self._timeStep, self._post_tau)

  -- Update time
  self._time = self._time + self._timeStep

  -- Perform spike callbacks
  --if self._spikeCallback then
  --  indices:apply(function(x) self._spikeCallback(x, self._time) end)
  --end
end

function SpikingReservoir:updateOutput(input)
  -- Input is a tensor that gets added to the potentials
  --self._potential:indexAdd(1, self._inputs, input * self._timeStep)
  --self._gGABAA:indexAdd(1, self._inputs, input * self._timeStep)
  --self._gGABAB:indexAdd(1, self._inputs, input * self._timeStep)
print(torch.sum(self._spikes))
  --self._outputHandler(target)
  self._outSpikes = torch.zeros(self._outputs:size())

  if self._cuda then
    self._outSpikes = self._outSpikes:cuda()
    input = input:cuda()
  end

  for i = 1, self._subSteps do
    self._inputHandler(input)
    self:timeStep()
  end

  self.output = self._outSpikes--self._potential:index(1, self._outputs)

  return self.output
end

function SpikingReservoir:updateGradInput(input, target)
  
end

function SpikingReservoir:cuda()
  self._cuda = true
  self._C = self._C:cuda()
  self._k = self._k:cuda()
  self._v_r = self._v_r:cuda()
  self._v_t = self._v_t:cuda()
  self._v_peak = self._v_peak:cuda()
  self._a = self._a:cuda()
  self._b = self._b:cuda()
  self._c = self._c:cuda()
  self._d = self._d:cuda()
  self._tau_x = self._tau_x:cuda()
  self._p = self._p:cuda()
  self._gAMPA_recep = self._gAMPA_recep:cuda()
  self._gNMDA_recep = self._gNMDA_recep:cuda()
  self._gGABAA_recep = self._gGABAA_recep:cuda()
  self._gGABAB_recep = self._gGABAB_recep:cuda()
  self._preTrace = self._preTrace:cuda()
  self._postTrace = self._postTrace:cuda()
  self._potential = self._potential:cuda()
  self._I = self._I:cuda()
  self._recovery = self._recovery:cuda()
  self._x = self._x:cuda()
  self._gAMPA = self._gAMPA:cuda()
  self._gNMDA = self._gNMDA:cuda()
  self._gGABAA = self._gGABAA:cuda()
  self._gGABAB = self._gGABAB:cuda()
  self._spikes = self._spikes:cuda()
  self._zeroes = self._zeroes:cuda()
  self._connections = self._connections:cuda()
  self._connections_d = self._connections_d:cuda()
  self._weights = self._weights:cuda()
  self._inputs = self._inputs:cuda()
  self._outputs = self._outputs:cuda()
end
