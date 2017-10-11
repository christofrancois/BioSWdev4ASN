local SparseCriterion = torch.class('nn.SparseCriterion', 'nn.Module')

function SparseCriterion:__init(sparsity, beta, scaler, running)
  if running == nil then
    running = true
  end

  self._sparsity = sparsity
  self._avgAct = nil
  self._samples = 0
  self._beta = beta or 1
  self._scaler = scaler or function(x) return x end
  self._ones = nil
  self.gradInput = torch.Tensor()
  self.output = torch.Tensor()
  self._running = running
end

function SparseCriterion:updateOutput(input, target)
  local batchSize = 1
  local original = input

  if self._running then

    if input:dim() == 2 then
      batchSize = input:size()[1]
      input = torch.sum(input, 1):squeeze()
    end

    if self._avgAct then
      self._avgAct = ( self._avgAct * self._samples
                   + input:clone():apply(self._scaler)
                   ) / (self._samples + batchSize)
      self._samples = self._samples + batchSize
    else
      self._avgAct = input:clone():apply(self._scaler)
      self._samples = batchSize
    end
    print(torch.norm(self._avgAct))
  end

  if self._avgAct then
    self._ones = self._ones or torch.ones(self._avgAct:size())
  end

--[[  self.output = self._beta
              * ( self._sparsity
                * torch.log(torch.cdiv(self._ones:clone():fill(self._sparsity), self._avgAct))
                + (1 - self._sparsity)
                * torch.log(torch.cdiv((self._ones - self._sparsity), (1 - self._avgAct))))
]]
  --self.output = torch.sum(self.output)

  input.THNN.Sigmoid_updateOutput(
    original:cdata(),
    self.output:cdata()
  )

  return self.output
end

function SparseCriterion:resize(input)
  if self.output:dim() == 1 then
    return input
  end

  local newsize = self.output:size()[1]
  return torch.repeatTensor(input, newsize, 1)
end

function SparseCriterion:updateGradInput(input, gradOutput)
  
  input.THNN.Sigmoid_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )

  if self._avgAct then

    local gradient = self._beta
                   * (torch.cdiv(self._ones - self._sparsity
                      , (1 - self._avgAct))
                   - torch.cdiv(self._ones:clone():fill(self._sparsity), self._avgAct))

    self.gradInput:add(
      torch.cmul(
        torch.cmul( self.output
                  , (self:resize(self._ones) - self.output))
        , self:resize(gradient)))
  end

  return self.gradInput
end
