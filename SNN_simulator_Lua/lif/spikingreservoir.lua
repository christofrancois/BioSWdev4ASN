require 'nn'

local SpikingReservoir, parent = torch.class('nn.SpikingReservoir','nn.Module')

-- Splits the given vector (self.vectorName) into _e/_i varants and adds them to self
function SpikingReservoir:splitVector(vectorName)
  self[vectorName .. '_e'] =
    self._nExcitatory > 0 and
      self[vectorName]:narrow(1, 1, self._nExcitatory)
    or torch.DoubleTensor()
  self[vectorName .. '_i'] =
    self._nInhibitory > 0 and
      self[vectorName]:narrow(1, self._nExcitatory + 1, self._nInhibitory)
    or torch.DoubleTensor()
end

function SpikingReservoir:__init(config)
  config = config or {}
  assert(torch.type(config) == 'table' and not config[1], 
    "Constructor requires key-value arguments")

  local defaultParams =
    { v_rest_e = -65
    , v_rest_i = -60
    , v_reset_e = -65
    , v_reset_i = -45
    , v_tresh_e = -52
    , v_tresh_i = -40
    , v_spike_e = 7
    , v_spike_i = -7
    , refrac_e = 5/1000
    , refrac_i = 2/1000
    , tc_theta = 1e7
    , a_plus = 0.1
    , a_minus = 0.1
    , A_plus = function(w) return (1 - w) end
    , A_minus = function(w) return w end
    }

  local args, nExcitatory, nInhibitory, connectivity, inputs, outputs, spikeCallback, timeStep, subSteps, modelParams = xlua.unpack(
      {config},
      'SpikingReservoir', 
      'A reservoir of spiking neurons',
      {arg='nExcitatory', type='number',
       help='Number of excitatory neurons'},
      {arg='nInhibitory', type='number',
       help='Number of inhibitory neurons'},
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
      {arg='modelParams', type='table', default=defaultParams,
       help='Neuron model parameters.'}
   )

  local nTotal = nExcitatory + nInhibitory
  assert(nTotal > 0, 'SpikingReservoir.__init: There must be at least one neuron!')

  self._nExcitatory = nExcitatory
  self._nInhibitory = nInhibitory
  self._nTotal = nTotal
  self._spikeCallback = spikeCallback
  self._timeStep = timeStep
  self._subSteps = subSteps
  self._preTrace = torch.zeros(nTotal)
  self._postTrace = torch.zeros(nTotal)
  self._potential = torch.zeros(nTotal)
  self:splitVector('_potential')
  self._potential_e:fill(modelParams.v_rest_e)
  self._potential_i:fill(modelParams.v_rest_i)
  self._ge = torch.zeros(nTotal)
  self._gi = torch.zeros(nTotal)
  self:splitVector('_ge')
  self:splitVector('_gi')
  self._theta = torch.zeros(nTotal)
  self._refrac = torch.zeros(nTotal)
  self:splitVector('_refrac')

  -- Construct connection and weight matrices
  self._connections = torch.lt( torch.rand(nTotal, nTotal)
                              , torch.ones(nTotal, nTotal) * connectivity)
  -- Zero the connections to self (diagonal)
  self._connections = self._connections - torch.diag(torch.diag(self._connections))
  
  if nExcitatory > 0 then
    self._weights = torch.rand(nExcitatory, nTotal)
  end
  if nInhibitory > 0 then
    if self._weights then
      self._weights = torch.cat(self._weights, torch.rand(nInhibitory, nTotal), 1)
    else
      self._weights = -torch.rand(nInhibitory, nTotal)
    end
  end

  self._weights:cmul(self._connections:double())
  self:splitVector('_weights')
  --self._weights_e = self._weights:narrow(1, 1, self._nExcitatory)
  --self._weights_i = self._weights:narrow(1, self._nExcitatory + 1, self._nInhibitory)
  self._inputs = inputs--:view(inputs:size(1),1)
  self._outputs = outputs--:view(outputs:size(1),1)
  self._params = modelParams
  self._time = 0
end

-- Calculate a single timestep
function SpikingReservoir:timeStep()
  local spikes_e = torch.gt(self._potential_e, self._params.v_tresh_e)
  local spikes_i = torch.gt(self._potential_i, self._params.v_tresh_i)
  local spikes = self._nExcitatory > 0 and spikes_e
  spikes = self._nInhibitory > 0
    and (spikes and torch.cat(spikes, spikes_i) or spikes_i)
    or spikes
  -- Reset spiked neurons
  self._potential_e:maskedFill(spikes_e, self._params.v_reset_e)
  self._potential_i:maskedFill(spikes_i, self._params.v_reset_i)

  -- Enter refractionary perioid
  self._refrac_e:maskedFill(spikes_e, self._params.refrac_e)
  self._refrac_i:maskedFill(spikes_i, self._params.refrac_i)

  -- Update potentials
  do
    local tmp = torch.zeros(self._nExcitatory, 1):fill(self._params.v_rest_e)
    self._potential_e:add(self._timeStep / 100
      , tmp
      - self._potential_e
      - torch.cmul(self._potential_e, self._ge_e + self._gi_e)
      - torch.mul(self._ge_e, 100)
      )
  end
  do
    local tmp = torch.zeros(self._nInhibitory, 1):fill(self._params.v_rest_i)
    self._potential_i:add(self._timeStep / 10
      , tmp
      - self._potential_i
      - torch.cmul(self._potential_i, self._ge_i + self._gi_i)
      - torch.mul(self._ge_i, 85)
      )
  end

  -- Update ge/gi
  self._gi:mul(self._gi, 1 - self._timeStep)
  self._ge:mul(self._ge, 1 - self._timeStep / 2)

  -- Add spikes to potentials
  local function nonzero(tensor)
    local tmp = tensor:nonzero()
    return tmp:reshape(tmp:nElement())
  end

  local indices = nonzero(spikes_e)
  if indices:nElement() > 0 then
    local tmp_e = self._weights_e:index(1, indices)
    tmp_e:mul(self._params.v_spike_e)
    self._potential:add(torch.sum(tmp_e, 1))
  end

  indices = nonzero(spikes_i)
  if indices:nElement() > 0 then
    local tmp_i = self._weights_i:index(1, indices)
    tmp_i:mul(self._params.v_spike_i)
    self._potential:add(torch.sum(tmp_i, 1))
  end

  -- Reset neurons in refractionary perioid
  local refrac_e = torch.gt(self._refrac_e, torch.zeros(self._nExcitatory, 1))
  local refrac_i = torch.gt(self._refrac_i, torch.zeros(self._nInhibitory, 1))
  self._potential_e:maskedFill(refrac_e, self._params.v_reset_e)
  self._potential_i:maskedFill(refrac_i, self._params.v_reset_i)

  -- Add spikes to weights
  indices = nonzero(spikes)
  if indices:nElement() > 0 then
    -- Incrase weights for presynaptic spikes
    self._weights = self._weights
      + self._weights:clone():apply(self._params.A_plus):cmul(
        torch.cmul(self._connections:double()
      , torch.repeatTensor(torch.cmul(spikes:double(), self._preTrace), self._nTotal, 1):t()))

  -- Decrease weights for postsynaptic spikes
    self._weights = self._weights
      - self._weights:clone():apply(self._params.A_minus):cmul(
        torch.cmul(self._connections:double()
      , torch.repeatTensor(torch.cmul(spikes:double(), self._postTrace), self._nTotal, 1)))


    -- Clamp weights to 0..1 range
    self._weights:clamp(0, 1)
  end

  -- Update pre/postsynaptic traces and
  -- add spikes to pre/postsynaptic traces
  self._preTrace = (self._preTrace * (1 - self._timeStep)):add(self._params.a_plus * spikes:double())
  self._postTrace = (self._postTrace * (1 - self._timeStep)):add(self._params.a_minus * spikes:double())
  
  -- Update refrac timers
  self._refrac:csub(self._timeStep)

  -- Update time
  self._time = self._time + self._timeStep

  indices:apply(function(x) self._spikeCallback(x, self._time) end)
end

function SpikingReservoir:STDP()
end

function SpikingReservoir:updateOutput(input)
  -- Input is a tensor that gets added to the potentials
 -- print(self._inputs)
  --print(input)
--print(self._potential)
  self._potential:indexAdd(1, self._inputs, input * self._timeStep)

  for i = 1, self._subSteps, 1 do
    self:timeStep()
  end

  self.output = self._potential:index(1, self._outputs)

  return self.output
end
