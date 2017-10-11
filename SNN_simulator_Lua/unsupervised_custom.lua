require 'nn'
require 'optim'
require 'gnuplot'
local aeviz = require 'aevisualizer'

local mnist_train = torch.load('/home/lehtolav/data/mnist/train.th7', 'ascii')
mnist_train[1] = mnist_train[1]:reshape(60000,28*28):double()
mnist_train[2] = mnist_train[2]:double() + 1

local inSize = 28*28
local hiddenSize = 1000
local outSize = 28*28
local trainN = 20000--60000
local epochs = 1
local batchSize = 50
local lambda = 0.001

local model = nn.Sequential()
local hiddenLayer = nn.Linear(inSize, hiddenSize)
model:add(nn.L1Penalty(lambda))
model:add(hiddenLayer)
local transfer = nn.Sigmoid()
model:add(transfer)
model:add(nn.L1Penalty(lambda))
local outputLayer = nn.Linear(hiddenSize, outSize)
model:add(outputLayer)

local criterion = nn.MSECriterion()

local sparsity = 0.05
local beta = 3
local resparse = 500
local sparsityVec = torch.zeros(hiddenSize):fill(sparsity)

local oldgrout = transfer.updateGradOutput

function addSparsity(sgrad)
  transfer.updateGradOutput = function(self, input, target)
    self:oldgrout(input, target)
    self.gradInput:add(torch.cmul(sgrad, torch.cmul(self.output, 1 - self.output)))
    return self.gradInput
  end
end

function calcSparseParam()
  model:forward(mnist_train[1])

  local avga = torch.sum(transfer.output, 1):squeeze() / trainN
  avga:clamp(1e-8,1 - 1e-8)
  local sgrad = beta * ( torch.cdiv(1 - sparsityVec
                         , (1 - avga))
                       - torch.cdiv(sparsityVec, avga))
  local sloss = torch.sum(sparsity * torch.log(torch.cdiv(sparsityVec, avga))
              + (1 - sparsity) * torch.log(torch.cdiv(1 - sparsityVec, 1 - avga)))
  return sloss, sgrad, avga
end

local x, dl_dx = model:getParameters()
local input
local sloss
--local printEvery = 100
local curLoss = 0
local resparseTimer = 0

function feval()
  dl_dx:zero()

  local output = model:forward(input, input)
  local loss = criterion:forward(output, input)
             + (lambda / 2) * torch.sum(
                 torch.pow(hiddenLayer.weight, 2) + torch.pow(outputLayer.weight, 2)
               )
  local gradInput = criterion:backward(output, input)
  model:backward(input, gradInput)

  loss = loss + sloss
  curLoss = loss

  return loss, dl_dx
end

for i = 1, epochs do
  print('Epoch ' .. i)
  local sgrad, avga
  
  local j = 1
  while j < trainN do
    if resparseTimer <= 0 then
      sloss, sgrad, avga = calcSparseParam()
      addSparsity(sgrad)
      resparseTimer = resparse
    end
    local k = math.min(batchSize, trainN - j + 1)
    input = mnist_train[1]:narrow(1, j, k)
    optim.lbfgs(feval, x)

    j = math.min(j + batchSize, trainN)
    resparseTimer = resparseTimer - k

    print('Example ' .. j .. '/' .. trainN)
    print('Current loss: ' .. curLoss)
    print(sloss, torch.norm(sgrad), torch.sum(avga)/avga:nElement())
  end
end

aeviz.visualize(hiddenLayer.weight, 28, nil, nil, false, true, 0)
gnuplot.figure()
aeviz.visualize(outputLayer.weight:t(), 28, nil, nil, false, true, 0)

local classifier = nn.Sequential()
classifier:add(nn.Linear(hiddenSize, 10))
classifier:add(nn.LogSoftMax())
local classCrit = nn.ClassNLLCriterion()
local target
model:forward(mnist_train[1])

local y, dl_dy = classifier:getParameters()

feval = function()
  dl_dy:zero()

  local output = classifier:forward(input, target)
  local loss = classCrit:forward(output, target)
  local gradInput = classCrit:backward(output, target)
  classifier:backward(input, gradInput)

  curLoss = loss
  return loss, dl_dy
end

print('Training the classifier')
batchSize = 20

for i = 1, epochs do
  print('Epoch ' .. i)
  
  local j = 1
  while j < trainN do
    local k = math.min(batchSize, trainN - j + 1)
    input = transfer.output:narrow(1, j, k)
    target = mnist_train[2]:narrow(1, j, k)
    optim.lbfgs(feval, y)

    j = math.min(j + batchSize, trainN)

    print('Example ' .. j .. '/' .. trainN)
    print('Current loss: ' .. curLoss)
  end
end

local mnist_test = torch.load('/home/lehtolav/data/mnist/test.th7', 'ascii')
local testN = 10000

mnist_test[1] = mnist_test[1]:reshape(10000,28*28):double()
mnist_test[2] = mnist_test[2]:double() + 1

model:forward(mnist_test[1])
classifier:forward(transfer.output)
local _, predictions = torch.max(classifier.output, 2)
local correct = torch.sum(torch.eq(predictions:squeeze(), mnist_test[2]:long()))
print(correct .. '/' .. testN .. ' classified correctly (' .. (correct / testN * 100) .. '%)')
