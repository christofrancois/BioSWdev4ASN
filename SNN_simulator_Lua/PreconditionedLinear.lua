local krylov = require 'krylov'
local PreconditionedLinear, parent = torch.class('nn.PreconditionedLinear', 'nn.Linear')

function PreconditionedLinear:updateGradInput(input, gradOutput)
  local P = krylov.pseudoInverse(self.weight)
  return parent:updateGradInput(input, P * gradOutput)
end
