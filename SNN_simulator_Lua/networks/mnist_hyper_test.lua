require 'nn'
require 'optim'

local utils = require 'utils'
local lsuv = require 'lsuv'
local augment = require 'augment'

local activation = nn.ReLU
local linear = nn.Linear

local channels = 1
local w1 = 28
local h1 = 28

local function id(x) return x end

local function makeConv(inFilters, kernel, outFilters, inSide, activation, normalize, padding, pooling, layerFunc)
  padding = padding or 1

  local layer = nn.Sequential()

  if pooling then
    layer:add(nn.SpatialMaxPooling(pooling, pooling, pooling, pooling))
  end

  layer:add(layerFunc(nn.SpatialConvolution(inFilters, outFilters, kernel, kernel, 1, 1, padding, padding)))

  if normalize then
    layer:add(nn.SpatialBatchNormalization(outFilters))
  end

  layer:add(activation())

  local outSide = inSide - kernel + 1 + padding * 2

  if pooling then
    outSide = inSide / 2 - kernel + 1 + padding * 2
  end
--print('conv', outFilters, outSide)
  return layer, outFilters, outSide
end

local function makeResidual(module)
  local newModule = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(module.module)
  concat:add(nn.Identity())
  newModule:add(concat)
  newModule:add(nn.CAddTable())
end

local function makeNetwork(inChannels, inSide, makeConv, makeLinear, moduleFunc, networkFunc)
  local modules = {}

  for _, v in pairs(makeConv) do
    local i = #modules + 1
    modules[i] = {}
    modules[i].inputs = inChannels * inSide * inSide
    modules[i].module, inChannels, inSide = v(inChannels, inSide)
    modules[i].outputs = inChannels * inSide * inSide
    modules[i].criterion = nn.MSECriterion()
    moduleFunc(modules[i])
  end

  inSide = inChannels * inSide * inSide
  modules[#modules].module:add(nn.View(inSide))

  for _, v in pairs(makeLinear) do
    local i = #modules + 1
    modules[i] = {}
    modules[i].inputs = inSide
    modules[i].module, inSide = v(inSide)
    modules[i].outputs = inSide
    modules[i].criterion = nn.MSECriterion()
    moduleFunc(modules[i])
  end

  return networkFunc(modules)
end

--[[
local lin2 = nn.Sequential()
lin2:add(renorm(linear(120, 84)))
lin2:add(activation())

local lin3 = renorm(linear(84, 1))
]]

local makeReLU = function(a) return function() return a(true) end end

local function initSpatial(layer)
end

local learnRates = { 1e-1, 1e-2, 1e-3, 1e-4 }

local mnist_train = torch.load('/home/lehtolav/data/mnist/train.th7', 'ascii')
mnist_train[1] = mnist_train[1]:reshape(60000, 1, 28, 28):double()
mnist_train[2] = mnist_train[2]:double() + 1

print('Augmenting training dataset')
local mnist_train_zca = augment.zca(mnist_train)
print('Augmented training dataset')

local moduleIter = utils.bindNonDet({ init = { 'lsuv', 'layerwise' }, style = { 'normal', 'static' }, augment = { 'none', 'zca' } },
  function(_, x)
    local normal = x.style == 'normal'
    local metametaParams =
      { activation =
        { nn.Tanh
        , makeReLU(nn.ReLU)
        --, makeReLU(nn.ReLU6)
        , nn.PReLU
        , nn.RReLU
        }
      , padding = normal and { 0 } or { 1 }
      , pooling = { false } --normal and { false, 2 } or { false }
      , normalize = { false, true }
      , kernel = normal and { 3, 5 } or { 3 }
      , layerFunc = x.init == 'lsuv' and { id } or { id, initSpatial }
      , moduleFunc = normal and { id } or { id, makeResidual }
      , networkFunc = x.init == 'lsuv' and { lsuv } or { id }
      , nConv = normal and { 2, 4, 6 } or { 2, 4, 8, 16, 32 }
      , nLin = normal and { 2, 4, 6 } or { 2, 4, 8, 16, 32 }
      }
    return utils.bindNonDet(metametaParams,
    function(_, y)
      local convFunc =
        function(inC, inS)
          return makeConv(inC, y.kernel, normal and inC + 2 or 32 , inS, y.activation, y.normalize, y.padding, y.pooling, y.layerFunc)
        end

      local linFunc =
        function(inS, outS)
          outS = outS or math.ceil(inS / 3 * 2)
          
          local layer = nn.Sequential()
--print(inS,outS)
          layer:add(nn.Linear(inS, outS))
          if y.normalize then
            layer:add(nn.BatchNormalization(outS))
          end
          layer:add(y.activation())
          return layer, outS
        end

      local networkFunc = function(modules)
          local lastInputs = modules[#modules].outputs
          modules[#modules + 1] =
            { module = linFunc(lastInputs, 10)
            , inputs = lastInputs
            , outputs = 10
            , criterion = nn.MSECriterion()
            }
          
          return y.networkFunc(
            { network = modules
            , batch = x.augment == 'none' and mnist_train[1]:narrow(1,1,10000) or mnist_train_zca[1]:narrow(1,1,10000)
            })
        end

      local module =
        makeNetwork(
            1
          , 28
          , utils.repTable(convFunc, y.nConv)
          , utils.repTable(linFunc, y.nLin)
          , y.moduleFunc
          , networkFunc
          )

      module.description = utils.tableMerge(x,y)

      return utils.bindNonDet(
        { trainset = 
            module.description.augment == 'none' and { mnist_train } or { mnist_train_zca }
        , finetuneCrit = { nn.CrossEntropyCriterion() }
        , method = { 'sgd', 'adam' }
        , state = utils.map(function(x) return {learningRate = x, momentum = 0.9} end, learnRates)
        , epochs = { 300 }
        , batchSize = { 1000 }
        , optStateFunc = { false, function(state, epoch, sample, loss)
                                    if epoch % 100 == 0
                                    then state.learningRate = state.learningRate / 10
                                    end end }
        , finetune = { --[['all', 'last',]] 'nopre' }
        },
      function(_, z)
--print(module.modules)
        module.description.method = z.method
        module.description.learningRate = z.state.learningRate
        module.description.batchSize = z.batchSize
        module.description.lRateDropoff = not (z.optStateFunc == false)
        module.description.finetuning = z.finetune
        return pairs({ utils.tableMerge(z, { modules = module }) })
      end
      )
    end
    )
  end
  )

local nConfig = 0
--[[
local configs = utils.bindNonDet(
        { trainset = 
            modules.description.augment == 'none' and { mnist_train } or { mnist_train_zca }
        , finetuneCrit = { nn.CrossEntropyCriterion() }
        , method = { 'sgd', 'adam' }
        , state = utils.map(function(x) return {learningRate = x, momentum = 0.9} end, learnRates)
        , epochs = { 300 }
        , batchSize = { 256, 1024 }
        , optStateFunc = { false, function(state, epoch, sample, loss)
                                    if epoch % 100 == 0
                                    then state.learningRate = state.learningRate / 10
                                    end end }
        , finetune = {'all', 'last', 'nopre'}
        }
      , function(k, x) return utils.iterMap(function(_, v) x.modules = v; return k, x end, moduleIter) end
      )]]

--[[
  utils.iterMap(function(k, modules)
      nConfig = nConfig + 1
      print('Generating config ' .. nConfig, modules)
      return k,
        { modules = modules
        , trainset = 
            modules.description.augment == 'none' and { mnist_train } or { mnist_train_zca }
        , finetuneCrit = { nn.CrossEntropyCriterion() }
        , method = { 'sgd', 'adam' }
        , state = utils.map(function(x) return {learningRate = x, momentum = 0.9} end, learnRates)
        , epochs = { 300 }
        , batchSize = { 256, 1024 }
        , optStateFunc = { false, function(state, epoch, sample, loss)
                                    if epoch % 100 == 0
                                    then state.learningRate = state.learningRate / 10
                                    end end }
        , finetune = {'all', 'last', 'nopre'}
        }
    end
  , moduleIter)]]

local mnist_test = torch.load('/home/lehtolav/data/mnist/test.th7', 'ascii')

mnist_test[1] = mnist_test[1]:reshape(10000, 1, 28, 28):double()
mnist_test[2] = mnist_test[2]:double() + 1

print('Augmenting testing dataset')
local mnist_test_zca = augment.zca(mnist_test)
print('Augmented testing dataset')

local validBatchSize = 256

local function getValidator(zca)
  return function(network)
    if network == 'predicate' then
      return true
    end

    local corrects = 0
    for i = 1, validBatchSize, 10000 do
      if zca then
        network:forward(mnist_test_zca[1]:narrow(1, i, math.min(i + validBatchSize, 10000)))
      else
        network:forward(mnist_test[1]:narrow(1, i, math.min(i + validBatchSize, 10000)))
      end

      local _, predictions = torch.max(network.output, 2)
      correct = correct
              + torch.sum(torch.eq( predictions:squeeze()
                                  , mnist_test[2]:narrow(1, i, math.min(i + validBatchSize, 10000)):long()))

    end

    network:clearState()

    return correct / 10000
  end
end

local config =
  { configs = moduleIter
  , validator = utils.nClassValidator
  , nValid = 0
  , tweakConfig = function(config)
      if config.method == 'sgd' then
        config.method = optim.sgd
      else
        config.method = function(f, x, s) optim.adam(f, x, { learningRate = s.learningRate }, s) end
      end
      config.validator = getValidator(config.modules.description.augment == 'zca')

      config.batchwise = 1000

      print(config.modules.description)

      return config
    end
  , postTrain = function(config, k, valAcc)
      torch.save('results/config' .. k .. '.cfg', { config = config, valAcc = valAcc }, 'ascii')
    end
  , trainset = mnist_train
  , testSet = mnist_test
  , onlyBest = true
  }

return config
