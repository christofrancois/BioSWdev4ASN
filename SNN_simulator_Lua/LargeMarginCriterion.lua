local LargeMarginCriterion, parent = torch.class('nn.LargeMarginCriterion', 'nn.Criterion')

function LargeMarginCriterion:__init(weights, lambda)
    parent.__init(self)

    if weights then
       assert(weights:dim() == 1, "weights input should be 1-D Tensor")
       self.weights = weights
    end

    self.lambda = lambda or 1

    self.target = torch.zeros(1):long()
end

function LargeMarginCriterion:__len()
   if (self.weights) then
      return #self.weights
   else
      return 0
   end
end

function LargeMarginCriterion:updateOutput(input, target)
   if type(target) == 'number' then
      if input:type() ~= 'torch.CudaTensor' then
         self.target = self.target:long()
      end
      self.target[1] = target
   elseif target:type() == 'torch.CudaTensor' then
      self.target = target
   else
      self.target = target:long()
   end
--[[
   local maxLH = nil

   if input.dim() == 1 then
     input = torch.reshape(input, 1, input:nElements())
   elseif input:dim() > 2 then
     error('Input dimension must be 1 or 2 (batch mode)')
   end

   if target > 1 then
     maxLH = torch.max(input:narrow(2, 1, target - 1))
   end

   if target < input:length() then
     local temp = torch.max(input:narrow(2, target + 1, input:size(2)))
     if not maxLH then
       maxLH = temp
     else
       maxLH = torch.cmax(maxLH, temp)
     end
   end

   self.output = self.lambda * (1 - (input[{{}, target}] - maxLH)) ^ 2]]

--(ff)' = 2ff'

   if input:dim() == 1 then
     input = torch.reshape(input, 1, input:nElements())
   elseif input:dim() > 2 then
     error('Input dimension must be 1 or 2 (batch mode)')
   end

   local nClass = input:size(2)
--print(target)
--print(input:size())
--print(input:gather(2, target:long():reshape(target:size(1),1)):size(), torch.sum(input, 2):size())
   local responses = input:gather(2, target:long():reshape(target:size(1),1))

   self.output = self.lambda * torch.sum(torch.pow(nClass - 1 - nClass * responses + torch.sum(input, 2), 2))  / (nClass - 1)

   return self.output
end

function LargeMarginCriterion:updateGradInput(input, target)
   if type(target) == 'number' then
      self.target[1] = target
   elseif target:type() == 'torch.CudaTensor' then
      self.target = target
   else
      self.target = target:long()
   end

   --self.gradInput:resizeAs(input):zero()
   if input:dim() == 1 then
     input = torch.reshape(input, 1, input:nElements())
   elseif input:dim() > 2 then
     error('Input dimension must be 1 or 2 (batch mode)')
   end

   local nClass = input:size(2)

   local responses = input:gather(2, target:long():reshape(target:size(1),1))

   self.gradInput = torch.repeatTensor(self.lambda * (nClass - 1 - nClass * responses + torch.sum(input, 2))  / (nClass - 1) * (nClass), 1, nClass)
   self.gradInput[{{},target}] = 0

   return self.gradInput
end

function LargeMarginCriterion:cuda()
  self.target = self.target:cuda()
  if self.weight then
    self.weight = self.weight:cuda()
  end
end
