local Sort, parent = torch.class('nn.Sort', 'nn.Module')

function Sort:__init()
  parent.__init(self)
  self.train = true
end

function Sort:updateOutput(input)
  if input:dim() > 2 then
    error('Input must be a 1D vector or a 2D batch!')
  end

  local sorted, perm = torch.sort(input)

  if self.train then
    self.perm = perm
  end

  self.output = sorted

  return self.output
end

function Sort:updateGradInput(input, gradOutput)
  if not self.gradInput or self.gradInput:dim() == 0
     or self.gradInput:size(1) ~= gradOutput:size(1) then
    self.gradInput = torch.zeros(gradOutput:size())
  end
  self.gradInput:scatter(gradOutput:dim(), self.perm, gradOutput)

  return self.gradInput
end

function Sort:evaluate()
  self.train = false
  self.perm = nil
end

function Sort:clearState()
  self.perm = nil
  return parent.clearState(self)
end
