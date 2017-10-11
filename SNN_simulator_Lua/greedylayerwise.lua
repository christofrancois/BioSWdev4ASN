--[[
greedyLayerwise implements the greedy layer-wise learning for deep neural networks.

requires the optim and unsup packages
]]
require 'optim'
require 'unsup'
local gl = {}
local utils = require 'utils'
local ProFi = require 'ProFi'
local optnet = require 'optnet'

local optnetOptions = { inplace = true, mode = 'training' }

function gl.greedyLayerwise(config, silent, profile, optimize)
  -- modules is an array of tables where each table contains the members:
  --   module: the actual module
  --   inputs: number of inputs
  --   outputs: number of outputs
  --   criterion: [optional] criterion used to train this layer of the network
  -- finetuneCrit is the criterion for fine-tuning
  silent = silent or false
  profile = profile or false
  optimize = optimize or false

  local verbose
  if silent then
    verbose = function() end
  else
    verbose = print
  end

  local printmodulo = 1000

  config = config or {}
  assert(torch.type(config) == 'table' and not config[1], 
    "Constructor requires key-value arguments")

  local args, modules, trainset, normalize,--[[learningrate, learningrateDecay,]] method, state, beta, lambda, finetuneCrit, epochs, batchSize, finetune, noise, cuda, optStateFunc, validator, batchwise = xlua.unpack(
      {config},
      'greedyLayerwise', 
      'Greedy layer-wise training for deep networks',
      {arg = 'modules', type = 'table',
       help = 'Array of modules and their input and output dimensions'},
      {arg = 'trainset', type = 'table',
       help = 'Training dataset'},
--      {arg = 'learningrate', type = 'number', default = 0.01,
--       help = 'Scale factor for learning'},
--      {arg = 'learningrateDecay', type = 'number', default = 0,
--       help = 'Decay (per epoch) for learning rate'},
      {arg = 'normalize', type = 'boolean', default = true,
       help = 'Whether or not to normalize data'},
      {arg = 'method', type = 'function', default = optim.sgd,
       help = 'Optimization method for training'},
      {arg = 'state', type = 'table', default = {},
       help = 'Initial state for optimization method'},
      {arg = 'beta', type = 'number', default = 3,
       help = 'Weight of sparsity parameter'},
      {arg = 'lambda', type = 'number', default = 1,
       help = 'Weight decay parameter'},
      {arg = 'finetuneCrit', type = 'nn.Criterion', default = nn.MSECriterion(),
       help = 'Criterion for finetuning.'},
      {arg = 'epochs', type = 'number', default = 5,
       help = 'How many times the data is repeated'},
      {arg = 'batchSize', type = 'number', default = 32,
       help = 'Number of examples shown to the network at a time'},
      {arg = 'finetune', type = 'string', default = 'all',
       help = 'What type of finetuning to perform'},
      {arg = 'noise', type = 'function', req = false,
       help = 'Given a matrix size (as LongStorage), generate a noise matrix to add in pretraining.'},
      {arg = 'cuda', type = 'boolean', default = false,
       help = 'Use CUDA.'},
      {arg = 'optStateFunc', type = 'function', req = false,
       help = 'function to call on optimizer state after every batch. Gets current loss as input'},
      {arg = 'validator', type = 'function', req = false,
       help = 'Validation function (must contain data as closure).'},
      {arg = 'batchwise', type = 'number', req = false,
       help = 'Make inputs for layers batchwise (not whole dataset at once)'}
  )

  verbose('Batch size: ' .. batchSize)

  local stackedAutoencoder = nn.Sequential()

  local trainInput = trainset[1]
  local trainTarget = trainset[2]
  local layerOutputs

  if normalize then
    local mean = trainInput:mean()
    local std = torch.std(trainInput)
    trainInput = (trainInput - mean) / std
  end

  if cuda then
    require 'cutorch'
    require 'cudnn'
    --trainInput = trainInput:cuda()
    --trainTarget = trainTarget:cuda() }4fj#J)Dr
    --stackedAutoencoder = cudnn.Sequential()
local net = nn.Sequential()
net:add(nn.SpatialConvolution(3,96,11,11,3,3))
net:add(nn.ReLU())
cudnn.convert(net, cudnn)
print(net)
  end

  local input
  local output
  local target

  --[[ Fix compatibility
  utils.hook(unsup.SparseAutoEncoder, 'updateOutput',
    function(f, self, i)
      f(self, i, target)
    end)]]
  --utils.preHook(nn.MSECriterion, 'updateOutput', function(_,i,t) print(i:size(), t:size()) end)

  local totalLoss = 0

  if profile then
    ProFi:checkMemory(nil, 'Before feval closure')
  end

  local function make_feval(model, crit)
    local x, dx = model:getParameters()
    return x, function()
      dx:zero()
--print(input:size())
--print(target:size())
--print(x:size())
--print(dx:size())
      output = model:updateOutput(input, target)
--print(output:size())
      model:updateGradInput(input, target)
      model:accGradParameters(input, target)
      totalLoss = totalLoss + output
      return output, dx
    end
  end

  if profile then
    ProFi:checkMemory(nil, 'After feval closure')
  end

  local nModules = #modules
  if finetune == 'last' then
    nModules = nModules - 1
  end

  for i = 1, nModules do
    verbose('Training layer ' .. i .. ' of ' .. #modules)

    if profile then
      ProFi:checkMemory(nil, 'Before creating autoencoder (' .. i .. ')')
    end

    local encoder = modules[i].module

    if cuda then
      cudnn.convert(encoder, cudnn)
    end

    if finetune ~= 'nopre' then
      local decoder = nn.Sequential()
      decoder:add(nn.View(modules[i].outputs))
      decoder:add(nn.Linear(modules[i].outputs, modules[i].inputs))
      local criterion = modules[i].criterion or finetuneCrit
      --local model = unsup.SparseAutoEncoder(encoder, decoder, beta, lambda, criterion)
      local model = unsup.AutoEncoder(encoder, decoder, beta, criterion)
      layerOutputs = torch.zeros(trainInput:size(1), modules[i].outputs)
      local tempstate = utils.deepCopyTable(state)
      totalLoss = 0

      decoder:training()
      encoder:training()

      if cuda then
        --decoder:cuda()
        --criterion:cuda()
        --encoder:cuda()
        --model:cuda()
        cudnn.convert(decoder, cudnn)
        cudnn.convert(criterion, cudnn)
        cudnn.convert(model, cudnn)
        --layerOutputs = layerOutputs:cuda()
      end

      if noise then
        trainInput = trainInput + noise(trainInput:size())
        if cuda then
          trainInput = trainInput:cuda()
        end
      end

      if profile then
        ProFi:checkMemory(nil, 'After creating autoencoder, before invoking make_feval (' .. i .. ')')
      end

      if optimize then
        local enc1 = optnet.countUsedMemory(encoder)
        local dec1 = optnet.countUsedMemory(decoder)
        local mod1 = optnet.countUsedMemory(model)
        optnet.optimizeMemory(encoder, trainInput[1], optnetOptions)
        optnet.optimizeMemory(decoder, encoder.output, optnetOptions)
        --optnet.optimizeMemory(model, trainInput[1], optnetOptions)
        local enc2 = optnet.countUsedMemory(encoder)
        local dec2 = optnet.countUsedMemory(decoder)
        local mod2 = optnet.countUsedMemory(model)
        local mega = 1024 * 1024
        verbose('Encoder: ' .. enc1.total_size/mega .. ' MBytes => '
              .. enc2.total_size/mega .. ' MBytes')
        verbose('Decoder: ' .. dec1.total_size/mega .. ' MBytes => '
              .. dec2.total_size/mega .. ' MBytes')
        verbose('Model: ' .. mod1.total_size/mega .. ' MBytes => '
              .. mod2.total_size/mega .. ' MBytes')
      end

      local params, feval = make_feval(model)

      if profile then
        ProFi:checkMemory(nil, 'After invoking make_feval (' .. i .. ')')
      end

      local shuffle = torch.randperm(trainInput:size(1)):long()

      for k = 1, epochs do

        verbose('Epoch ' .. k .. ' of ' .. epochs)

        for j = 1, trainInput:size(1), batchSize do
          input = trainInput:index(1, shuffle[{{j, math.min(j + batchSize - 1, trainInput:size(1))}}])

          if cuda then
            input = input:cuda()
          end

          target = input:clone()

          method(feval, params, tempstate)

          if (j-1) % printmodulo == 0 then
            verbose( 'Example ' .. j .. ' of ' .. trainInput:size(1)
                  .. ', avgLoss: ' .. totalLoss / (j + (k-1) * trainInput:size(1)))
          end

          collectgarbage()
        end
      end

      if profile then
        ProFi:checkMemory(nil, 'Before creating input for next layer (' .. i .. ')')
      end

      decoder:evaluate()
      encoder:evaluate()

      if batchwise then
        layerOutputs = nil
        local outs = {}
        local batchN = 1
        for j = 1, trainInput:size(1), batchwise do
          input = trainInput:index(1, shuffle[{{j, math.min(j + batchwise - 1, trainInput:size(1))}}])

          if cuda then
            input = input:cuda()
          end

--          layerOutputs = layerOutputs and torch.cat(layerOutputs, encoder:forward(input), 1)
--                      or encoder:forward(input)
          outs[batchN] = encoder:forward(input)
          batchN = batchN + 1
        end
        layerOutputs = torch.cat(outs, 1)
      else
        layerOutputs = encoder:forward(trainInput)
      end

--    if i < #modules then
        trainInput = layerOutputs:clone()
  --  end

      if profile then
        ProFi:checkMemory(nil, 'After making input, before stacking (' .. i .. ')')
      end

      encoder:clearState()
    --criterion:clearState()

    end -- if finetune ~= 'nopre'

    stackedAutoencoder:add(encoder)

    if profile then
      ProFi:checkMemory(nil, 'After stacking (' .. i .. ')')
    end
  end

  layerOutputs = nil

  if optimize then
    local mem1 = optnet.countUsedMemory(stackedAutoencoder)
    optnet.optimizeMemory(stackedAutoencoder, trainset[1][1], optnetOptions)
    local mem2 = optnet.countUsedMemory(stackedAutoencoder)
    verbose('Overall: ' .. mem1.total_size/mega .. ' MBytes => '
          .. mem2.total_size/mega .. ' MBytes')
  end

  local valAcc = torch.zeros(epochs)

  if finetune ~= 'none' then
    verbose('Finetuning')

    local loss
    local totalLoss = 0

    make_feval = function(model, crit)
      local x, dx = model:getParameters()
      return x, function()
        dx:zero()
        output = model:forward(input)
        loss = crit:forward(output, target)
        local grad = crit:backward(output, target)
        model:backward(input, grad)
        return loss, dx
      end
    end

    if cuda then
      --stackedAutoencoder:cuda()
      --finetuneCrit:cuda()
      cudnn.convert(stackedAutoencoder, cuddn)
      cudnn.convert(finetuneCrit, cudnn)
      if finetune == 'last' then
        --modules[#modules].module:cuda()
        cudnn.convert(modules[#modules].module, cudnn)
      end
    end

    local params, feval
    if finetune == 'all' or finetune == 'nopre' then
      trainInput = trainset[1]
      params, feval = make_feval(stackedAutoencoder, finetuneCrit)
    elseif finetune == 'last' then
      params, feval = make_feval(modules[#modules].module, finetuneCrit)
      stackedAutoencoder:add(modules[#modules].module)
    else
      verbose('unknown type of finetuning "' .. finetune .. '"')
      return stackedAutoencoder
    end

    local tempstate = utils.deepCopyTable(state)

    stackedAutoencoder:training()

    for j = 1, epochs do
      totalLoss = 0
      local shuffle = torch.randperm(trainInput:size(1)):long()
      verbose('Epoch ' .. j .. ' of ' .. epochs)
      for i = 1, trainInput:size(1), batchSize do
        if (i-1) % printmodulo == 0 then
          verbose( 'Example ' .. i .. ' of ' .. trainInput:size(1)
                .. ', avgLoss: ' .. totalLoss / i)
        end
        input = trainInput:index(1, shuffle[{{i, math.min(i + batchSize, trainInput:size(1))}}])
        target = trainTarget:index(1, shuffle[{{i, math.min(i + batchSize, trainInput:size(1))}}])

        if optStateFunc then
          optStateFunc(tempstate, j, i, totalLoss)
        end

        if cuda then
          input = input:cuda()
          target = target:cuda()
        end

        method(feval, params, tempstate)
        totalLoss = totalLoss + loss
      end
      
      if validator then
        stackedAutoencoder:evaluate()
        valAcc[j] = validator(stackedAutoencoder)
        stackedAutoencoder:training()
        verbose('Validation error: ' .. (1 - valAcc[j]))
      end
    end
  end

  --print(stackedAutoencoder:forward(trainInput))

  -- Clear internal state (temporary variables) of the network
  -- Makes the network much smaller in memory
  stackedAutoencoder:clearState()
  stackedAutoencoder:evaluate()

  return stackedAutoencoder, valAcc
end

-- Split some datas off of a dataset
function gl.splitSet(dataset, n)
  if n == 0 then
    return nil, dataset
  end

  local ndata = dataset[1]:size(1)
  local shuffle = torch.randperm(ndata):long()

  local dset1, dset2 = {}, {}
  dset1[1] = dataset[1]:index(1, shuffle[{{1, n}}])
  dset1[2] = dataset[2]:index(1, shuffle[{{1, n}}])
  dset2[1] = dataset[1]:index(1, shuffle[{{n + 1, ndata}}])
  dset2[2] = dataset[2]:index(1, shuffle[{{n + 1, ndata}}])
  return dset1, dset2
end

-- Treat config as a "non-deterministic table" and test each hyperparameter combination
function gl.hyperGreedyLayerwise(config, silent, profile, optimize)
  local args, configs, validator, nValid, tweakConfig, trainset, testSet, skipConfig, onlyBest, postTrain = xlua.unpack(
      {config},
      'greedyLayerwise', 
      'Greedy layer-wise training for deep networks',
      {arg = 'configs', type = 'table',
       help = 'Table or iterator to choose configs from'},
      {arg = 'validator', type = 'function', 
       help = 'Function taking a trained net and some data. Should return some measure of accuracy (larger value = better).'},
      {arg = 'nValid', type = 'number', 
       help = 'The number of validation datapoints to choose from the dataset'},
      {arg = 'tweakConfig', type = 'function', default = function(x) return x end,
       help = 'Each generated config is passed through this function (only when validating, see next).'},
      {arg = 'trainset', type = 'table', req = false,
       help = 'If given, the best performed network is retrained (with its untweaked configs) using this dataset. Also, configs can define an empty dataset to use this global set instead. The purpose is to allow the tweaking to modify the config such that validations are done on a smaller subset of data or for smaller amount of epochs to speed up the process.'},
      {arg = 'testSet', type = 'table', req = false,
       help = 'Testing dataset for the final network.'},
      {arg = 'skipConfig', type = 'number', req = false,
       help = 'Number of configs to skip.'},
      {arg = 'onlyBest', type = 'boolean', default = false,
       help = 'Only keep the best config.'},
      {arg = 'postTrain', type = 'function', req = false,
       help = 'Function to call post training with config, its number and validation loss.'}
  )

  local verbose = print

  if silent then
    verbose = function() end
  end

  if type(testSet) == 'function' then
    testSet = testSet()
  end

  local results = {}
  local bestk = nil
  local iterator = type(configs) == 'table' and utils.nonDeterministic(configs) or configs

  for k, config in iterator do
--print(config)
--print(config.modules)
    if skipConfig and (type(skipConfig) == 'function' and skipConfig(k, config) or type(skipConfig) == 'number' and skipConfig >= k) then
      verbose('Skipping config #' .. k)
    else
      verbose('Trying config #' .. k)

      results[k] = {}
      results[k].config = utils.deepCopyTable(config)
      -- Allow the training set to be garbage collected after training
      results[k].config.trainset = nil

      if #config.trainset == 0 and trainset then
        config.trainset = trainset
      end
      config = tweakConfig(config)

      local valid, train = gl.splitSet(config.trainset, nValid)
      local confValidator = config.validator or validator

      config.trainset = train
      config.validator = function(net) return confValidator(net, valid) end
      local trained, valAcc = gl.greedyLayerwise(config, silent, profile, optimize)

      local accuracy = confValidator(trained, valid)
      trained:clearState()

      if not trainset then
        results[k].trained = trained
      end

      results[k].accuracy = accuracy

      if postTrain then
        postTrain(results[k], k, valAcc)
      end

      if bestk == nil then
        bestk = k
      elseif results[k].accuracy > results[bestk].accuracy then
        if onlyBest then
          results[bestk] = nil
        end
        bestk = k
      elseif onlyBest then
        results[k] = nil
      end

      collectgarbage()

      verbose('Config #' .. k .. ' accuracy was ' .. accuracy .. '.')
      verbose('(Best: ' .. results[bestk].accuracy .. ' / #' .. bestk .. ')')
    end
  end

  verbose('Saving config data...')
  torch.save('bestconf.cfg', results, 'ascii')

  verbose('Best config:')
  verbose(results[bestk])

  local finalNet = results[bestk].trained

  if trainset then
    verbose('Retraining with best settings...')
    results[bestk].config.trainset = trainset
    finalNet = gl.greedyLayerwise(results[bestk].config, silent, profile, optimize)
  end

  local testAccuracy

  if testSet then
    testAccuracy = validator(finalNet, testSet)
    verbose('Testing accuracy: ' .. testAccuracy)
  end

  -- Repeat config specs
  verbose('Used config:')
  verbose(results[bestk])

  return finalNet, results, testAccuracy
end

return gl
