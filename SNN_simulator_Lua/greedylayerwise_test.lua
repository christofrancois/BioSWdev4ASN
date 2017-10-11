require 'nn'
require 'optim'
require 'gnuplot'
require 'unsup'
local aeviz = require 'aevisualizer'
local gl = require 'greedylayerwise'
local utils = require 'utils'
--[[
nn.tables = nn.tables or {}

function nn.tables.full(nin, nout)
   local ft = torch.Tensor(nin*nout,2)
   local p = 1
   for j=1,nout do
      for i=1,nin do
	 ft[p][1] = i
	 ft[p][2] = j
	 p = p + 1
      end
   end
   return ft
end]]

local mnist_train = torch.load('/home/lehtolav/data/mnist/train.th7', 'ascii')

local inSide = 28
local inSize = inSide*inSide
local hiddenSize = 78
local outSize = inSide*inSide
local trainN = 60000--60000
local epochs = 1
local batchSize = 50
local lambda = 0.01
local lrate = 1e-5
local lrdecay = 1e-12

mnist_train[1] = mnist_train[1]:reshape(60000,28*28):double():narrow(1, 1, trainN)
mnist_train[2] = mnist_train[2]:double():narrow(1, 1, trainN) + 1

-- Normalize data
local mean = mnist_train[1]:mean()
local std = torch.std(mnist_train[1])
mnist_train[1] = (mnist_train[1] - mean) / std

local beta = 1
local lambda = 1
local hiddenLayer = nn.Linear(inSize, hiddenSize)

local hiddenModule = nn.Sequential()
hiddenModule:add(hiddenLayer)
hiddenModule:add(nn.Sigmoid())

local outModule = nn.Sequential()
outModule:add(nn.Linear(hiddenSize, 10))

local function makeLayer(tr, inputs, outputs, criterion)
  local seq = nn.Sequential()
  tr = tr or 'Sigmoid'
  inputs = inputs or hiddenSize
  outputs = outputs or hiddenSize
  criterion = criterion or nn.MSECriterion()
  seq:add(nn.Linear(inputs, outputs))
  seq:add(nn[tr]())
  return { module = seq, inputs = inputs, outputs = outputs, criterion = criterion }
end

local modules =
  { { module = hiddenModule, inputs = inSize, outputs = hiddenSize, criterion = nn.MSECriterion() }
  , { module = outModule, inputs = hiddenSize, outputs = 10, criterion = nn.MSECriterion() }
  }

local autoEncoder = gl.greedyLayerwise
  { modules = modules
  , trainset = mnist_train
  , finetuneCrit = nn.CrossEntropyCriterion()
  , beta = beta
  , lambda = lambda
  --, method = optim.adam --optim.sgd
  , method = function(f, x, s)
      local config =
        { learningRate = 0.001
        , beta1 = 0.9
        , beta2 = 0.999
        , epsilon = 1e-8
        }
      optim.adam(f, x, config, s) --optim.sgd
    end
  , state =
    { learningRate = lrate
    , learningRateDecay = lrdecay
    , lineSearch = nil --optim.lswolfe
    }
  , epochs = 1
  , finetune = 'last'
  }

--gnuplot.figure()
--aeviz.visualize(outputLayer.weight:t(), 28, nil, nil, false, true, 0)

local mnist_test = torch.load('/home/lehtolav/data/mnist/test.th7', 'ascii')
local testN = 10000

mnist_test[1] = mnist_test[1]:reshape(10000,28*28):double()
mnist_test[2] = mnist_test[2]:double() + 1

autoEncoder:forward(mnist_test[1])
--print(autoEncoder.output)
local _, predictions = torch.max(autoEncoder.output, 2)
local correct = torch.sum(torch.eq(predictions:squeeze(), mnist_test[2]:long()))
print(correct .. '/' .. testN .. ' classified correctly (' .. (correct / testN * 100) .. '%)')

for k, v in pairs(modules) do
  local params = v.module:getParameters()
  local weight = params[1]
  if weight and type(weight) ~= 'number' then
    gnuplot.figure()
    local size = weight:size(2)
    local width = math.floor(math.sqrt(size) + 0.5)
    local height = size / width

    while height ~= math.floor(height) do
      width = width - 1
      height = size / width
    end

    aeviz.visualize(weight, width, height, nil, false, true, 0)
  end
end
