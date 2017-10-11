require 'nn'
require 'optim'
require 'gnuplot'
require 'unsup'
local aeviz = require 'aevisualizer'
local gl = require 'greedylayerwise'
local utils = require 'utils'
local ProFi = require 'ProFi'

require 'PreconditionedLinear'

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

local profile = true

if profile then
  ProFi:start()
end

local mnist_train = torch.load('/home/lehtolav/data/mnist/train.th7', 'ascii')

local inSide = 28
local inSize = inSide*inSide
local hiddenSize = 78
local outSize = inSide*inSide
local trainN = 60000--60000
local epochs = 1
local batchSize = 50
local lrate = 1e-5
local lrdecay = 1e-12

mnist_train[1] = mnist_train[1]:reshape(60000,28*28):double():narrow(1, 1, trainN)
mnist_train[2] = mnist_train[2]:double():narrow(1, 1, trainN) + 1
--error(mnist_train[2]:size())
--[[local krylov = require 'krylov'
gnuplot.imagesc(mnist_train[1][1]:reshape(28,28))
gnuplot.figure()
gnuplot.imagesc(krylov.krylov(mnist_train[1], 28*28))

error()
]]

-- Normalize data
local mean = mnist_train[1]:mean()
local std = torch.std(mnist_train[1])
mnist_train[1] = (mnist_train[1] - mean) / std

local beta = 1
local lambda = 1
local hiddenLayer = nn.Linear(inSize, hiddenSize)

local function renorm(layer)
  layer.weight = (layer.weight + 1) * (0.8/2) + 0.2
  return layer
end

local activation = nn.ReLU
local linear = nn.Linear
local snormal = nn.SpatialBatchNormalization
local normal = nn.BatchNormalization

local conv1 = nn.Sequential()
local filtersIn, filtersOut = 1, 6
local kernelSize = 5
conv1:add(nn.View(1, inSide, inSide))
conv1:add(renorm(nn.SpatialConvolution(filtersIn, filtersOut, kernelSize, kernelSize)))
conv1:add(activation())

local conv2 = nn.Sequential()
conv2:add(nn.SpatialMaxPooling(2,2,2,2))
conv2:add(renorm(nn.SpatialConvolution(6, 16, 5, 5)))
conv2:add(activation())

local lin1 = nn.Sequential()
lin1:add(nn.SpatialMaxPooling(2,2,2,2))
lin1:add(nn.View(16*4*4))
lin1:add(renorm(linear(16*4*4, 120)))
lin1:add(activation())

local lin2 = nn.Sequential()
lin2:add(renorm(linear(120, 84)))
lin2:add(activation())

local lin3 = renorm(linear(84, 10))

--print(lin3.weight); error()
--utils.preHook(nn.Linear, 'updateOutput', function(self, i)
--    print(self.weight:size())
--    print(i:size())
--  end)

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
  { { module = conv1
    , inputs = inSize
    , outputs = filtersOut*(inSide - kernelSize + 1)^2
    , criterion = nn.MSECriterion()
    }
  , { module = conv2
    , inputs = filtersOut*(inSide - kernelSize + 1)^2
    , outputs = 16*((inSide - kernelSize + 1)/2 - 5 + 1)^2
    , criterion = nn.MSECriterion()
    }
  , { module = lin1
    , inputs = 16*8*8
    , outputs = 120
    , criterion = nn.MSECriterion()
    }
  , { module = lin2
    , inputs = 120
    , outputs = 84
    , criterion = nn.MSECriterion()
    }
  , { module = lin3
    , inputs = 84
    , outputs = 10
    , criterion = nn.MSECriterion()
    }
  }

local function makeMethod(config)
  return function(f, x, s)
    if f == 'getConfig' then
      return config
    else
      optim.adam(f, x, config, s)
    end
  end
end

local metconfigs =
  { learningRate = { 1e-1, 1e-2, 1e-3, 1e-4, 1e-5 }
  , beta1 = { 0.9, 0.9*1.01, 0.9*1.02, 0.9*0.99, 0.9*0.98 }
  , beta2 = { 0.999, 0.999 + (0.001*0.50), 0.999 + (0.001*0.25), 0.999*0.99, 0.999*0.98 }
  , epsilon = { 1e-8 }
  }

local configs =
  { modules = { modules }
  , trainset = { mnist_train }
  , finetuneCrit = { nn.CrossEntropyCriterion() }
  --, method = optim.sgd
  , method = utils.map(makeMethod, utils.nonDeterministic(metconfigs, false))
  , state = { {} }--[[utils.nonDeterministic(
    { learningRate = lrate
    , learningRateDecay = lrdecay
    }, false)]]
  , epochs = { 300 }
  , batchSize = { 16, 32, 64, 128 }
  , finetune = {'all', 'last'}
  }

local mnist_test = torch.load('/home/lehtolav/data/mnist/test.th7', 'ascii')
local testN = 10000

mnist_test[1] = mnist_test[1]:reshape(10000,28*28):double()
mnist_test[2] = mnist_test[2]:double() + 1
local config =
  { configs = configs
  , validator = utils.nClassValidator
  , nValid = 10000
  , tweakConfig = function(config)
      config.epochs = 5
      config.trainset[1] = config.trainset[1]:narrow(1, 1, 15000)
      config.trainset[2] = config.trainset[2]:narrow(1, 1, 15000)
      print('Settings:')
      local metconf = config.method('getConfig')
      print('Learningrate: ' .. metconf.learningRate)
      print('beta1: ' .. metconf.beta1)
      print('beta2: ' .. metconf.beta2)
      print('epsilon: ' .. metconf.epsilon)
      print('Finetuning: ' .. config.finetune)
      return config
    end
  , trainset = mnist_train
  , testSet = mnist_test
  }

local autoEncoder = gl.hyperGreedyLayerwise(config, false, false, false)
--f(x)e = f(x + e)e
--F(x+e) - F(x) = f(x+e)e
--gnuplot.figure()
--aeviz.visualize(outputLayer.weight:t(), 28, nil, nil, false, true, 0)

autoEncoder:forward(mnist_test[1])
--print(autoEncoder.output)
local _, predictions = torch.max(autoEncoder.output, 2)
local correct = torch.sum(torch.eq(predictions:squeeze(), mnist_test[2]:long()))
print(correct .. '/' .. testN .. ' classified correctly (' .. (correct / testN * 100) .. '%)')

-- To normalize, use same mean and std as for training data,
-- so that the test data is brought to the same feature space
mnist_test[1] = (mnist_test[1] - mean) / std
print('after normalizing...')
_, predictions = torch.max(autoEncoder.output, 2)
correct = torch.sum(torch.eq(predictions:squeeze(), mnist_test[2]:long()))
print(correct .. '/' .. testN .. ' classified correctly (' .. (correct / testN * 100) .. '%)')

if profile then
  ProFi:stop()
  ProFi:writeReport('profile.txt')
end

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
