require 'nn'

local SpikingReservoir, parent = torch.class('nn.SpikingReservoir','nn.Module')

--[[ Splits the given vector (self.vectorName) into _e/_i varants and adds them to self
function SpikingReservoir:splitVector(vectorName)
  self[vectorName .. '_e'] =
    self._nExcitatory > 0 and
      self[vectorName]:narrow(1, 1, self._nExcitatory)
    or torch.DoubleTensor()
  self[vectorName .. '_i'] =
    self._nInhibitory > 0 and
      self[vectorName]:narrow(1, self._nExcitatory + 1, self._nInhibitory)
    or torch.DoubleTensor()
end]]

function SpikingReservoir:__init(config)
  config = config or {}
  assert(torch.type(config) == 'table' and not config[1], 
    "Constructor requires key-value arguments")

  local defaultParams =
    { a_minus = 1
    , a_plus = 1
    , A_minus = function(x) return x end
    , A_plus = function(x) return (1 - x) end
    }

  local args, neurons, connectivity, inputs, outputs, spikeCallback, timeStep, subSteps, params, n = xlua.unpack(
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
      {arg='spikeCallback', type='function',
       help='Function to call on spikes'},
      {arg='timeStep', type='number',
       help='Length (in time) of a simulation step'},
      {arg='subSteps', type='number',
       help='Number of steps in a simgle simulation step'},
      {arg='params', type='table', default=defaultParams,
       help='Parameters for STDP'},
      {arg='n', type='number', default=1,
       help='Number of synapses between neurons. Simulated as simply a factor on synaptic currents.'}
  )

  -- These are the scales of the parameters in the system
  -- These are a recent addition; feel free to try resetting them to 1
  -- (i.e. disabling them)
  self._faradScale = 1e-12  -- picofarads
  self._voltScale = 1e-3    -- millivolts
  self._ampereScale = 1e-12 -- picoamperes
  self._timeScale = 1e-3    -- milliseconds
  local kilo = 1e3          -- used to turn kHz to Hz
  --]]
  --[[
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
  self._timeStep = timeStep
  self._subSteps = subSteps
  self._preTrace = torch.zeros(nTotal)
  self._postTrace = torch.zeros(nTotal)
  self._potential = torch.zeros(nTotal):fill(-65) * self._voltScale
  self._recovery = torch.zeros(nTotal)
  self._x = torch.ones(nTotal, nTotal)
  self._gAMPA = torch.zeros(nTotal)
  self._gNMDA = torch.zeros(nTotal)
  self._gGABAA = torch.zeros(nTotal)
  self._gGABAB = torch.zeros(nTotal)
  self._spikes = torch.zeros(nTotal)
  
  -- Construct connection and weight matrices
  self._connections = torch.lt( torch.rand(nTotal, nTotal)
                              , torch.ones(nTotal, nTotal) * connectivity)
  -- Zero the connections to self (diagonal)
  self._connections = self._connections - torch.diag(torch.diag(self._connections))
  
  self._weights = torch.ones(nTotal, nTotal)--torch.rand(nTotal, nTotal)
  self._weights:cmul(self._connections:double())

  self._inputs = inputs
  self._outputs = outputs
  self._params = params
  self._n = n
  self._time = 0

  -- Reverse potentials and other constants
  self._E_AMPA = 80 * self._voltScale
  self._AMPA_divisor = 60 * self._voltScale
  self._E_GABAA = 70 * self._voltScale
  self._E_GABAB = 90 * self._voltScale

  -- Time constants
  self._gAMPA_tau = 5 * self._timeScale
  self._gNMDA_tau = 150 * self._timeScale
  self._gGABAA_tau = 6 * self._timeScale
  self._gGABAB_tau = 150 * self._timeScale
end

-- Calculate a single timestep
function SpikingReservoir:timeStep(input)
  self._spikes = torch.gt(self._potential, self._v_peak)
  local spikes_d = self._spikes:double() -- Store type conversion for later use

  -- Reset spiked neurons
  self._potential:maskedCopy(self._spikes, self._c[self._spikes])
  self._recovery:add(torch.cmul(spikes_d, self._d))

  -- Update potentials
  local tmp = torch.pow((self._potential + self._E_AMPA) / self._AMPA_divisor, 2)
  local vmv_r = self._potential - self._v_r
  -- This is the synaptic current
  local I = torch.cmul(
              self._potential
            , self._gAMPA + torch.cmul(self._gNMDA, torch.cdiv(tmp, 1 + tmp))
            )
            + torch.cmul(self._gGABAA, self._potential + self._E_GABAA)
            + torch.cmul(self._gGABAB, self._potential + self._E_GABAB)

  I:indexAdd(1, self._inputs, -input)

  -- Dendritic current is not simulated

  -- Here, the current I is added to the potential in equations in sources, but I is
  -- the sum of negations of the dendritic and synaptic currents.
  -- For performance reasons we do not want to calculate negations in the previous formula.
  self._potential:add(self._timeStep
    , torch.cdiv(
        torch.cmul(
          self._k
        , torch.cmul(
            vmv_r
          , self._potential - self._v_t
          )
        )
      - self._recovery
      - I * self._ampereScale
      , self._C
      )
    )

  -- Update recovery
  self._recovery:add(self._timeStep
    , torch.cmul(self._a
    , torch.cmul(self._b, vmv_r) - self._recovery)
    )

  -- Add spikes to conductance factor
  local p_spike_t = torch.repeatTensor(torch.cmul(spikes_d, self._p), self._nTotal, 1):t()
  self._x:cmul(1 - torch.cmul(self._connections:double(), p_spike_t))

  -- Update conductance factor
  self._x:add(self._timeStep, torch.cdiv(1 - self._x, self._tau_x))

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
  self._gAMPA:add(-self._timeStep / self._gAMPA_tau, self._gAMPA)
  self._gNMDA:add(-self._timeStep / self._gNMDA_tau, self._gNMDA)
  self._gGABAA:add(-self._timeStep / self._gGABAA_tau, self._gGABAA)
  self._gGABAB:add(-self._timeStep / self._gGABAB_tau, self._gGABAB)

  -- Add spikes to weights
  local function nonzero(tensor)
    local tmp = tensor:nonzero()
    -- Result is n x 1 tensor: reshape it to a length n vector
    -- squeeze() doesn't work as a scalar tensor (n=1) gets squeezed to a number
    return tmp:reshape(tmp:nElement())
  end

  indices = nonzero(self._spikes)
  if indices:nElement() > 0 then
    -- Incrase weights for presynaptic spikes
    -- A_plus(x) -> 1 - x
    self._weights:add(
        self._weights:clone():apply(self._params.A_plus):cmul(
        torch.cmul(self._connections:double()
      , torch.addr(torch.zeros(self._nTotal, self._nTotal)
        , self._preTrace, spikes_d))))
--          torch.repeatTensor(torch.cmul(spikes_d, self._preTrace), self._nTotal, 1):t()
--        , torch.repeatTensor(spikes_d, self._nTotal, 1)))))

    -- Decrease weights for postsynaptic spikes
    -- A_minus(x) -> x
    self._weights = self._weights
      - self._weights:clone():apply(self._params.A_minus):cmul(
        torch.cmul(self._connections:double()
      , torch.addr(torch.zeros(self._nTotal, self._nTotal), spikes_d, self._postTrace)))
    --torch.repeatTensor(torch.cmul(spikes_d, self._postTrace), self._nTotal, 1)))

    -- Clamp weights to [0,1] range
    self._weights:clamp(0, 1)
  end

  -- Update pre/postsynaptic traces and
  -- add spikes to pre/postsynaptic traces
  -- Perhaps we need only one trace if a+ = a-
  self._preTrace:add(self._timeStep, self._params.a_plus * spikes_d - self._preTrace)
  self._postTrace:add(self._timeStep, self._params.a_minus * spikes_d - self._postTrace)

  -- Update time
  self._time = self._time + self._timeStep

  -- Perform spike callbacks
  indices:apply(function(x) self._spikeCallback(x, self._time) end)
end

function SpikingReservoir:updateOutput(input)
  -- Input is a tensor that gets added to the potentials
  --self._potential:indexAdd(1, self._inputs, input * self._timeStep)
  --self._gGABAA:indexAdd(1, self._inputs, input * self._timeStep)
  --self._gGABAB:indexAdd(1, self._inputs, input * self._timeStep)

  for i = 1, self._subSteps do
    self:timeStep(input)
  end

  self.output = self._potential:index(1, self._outputs)

  return self.output
end
