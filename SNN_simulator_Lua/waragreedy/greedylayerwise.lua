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

  config = config or {}
  assert(torch.type(config) == 'table' and not config[1], 
    "Constructor requires key-value arguments")

  local args, modules, trainset, normalize,--[[learningrate, learningrateDecay,]] method, state, beta, lambda, finetuneCrit, epochs, batchSize, finetune = xlua.unpack(
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
       help = 'What type of finetuning to perform'}
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
    local decoder = nn.Sequential()
    decoder:add(nn.View(modules[i].outputs))
    decoder:add(nn.Linear(modules[i].outputs, modules[i].inputs))
    local criterion = modules[i].criterion or finetuneCrit
    --local model = unsup.SparseAutoEncoder(encoder, decoder, beta, lambda, criterion)
    local model = unsup.AutoEncoder(encoder, decoder, beta, criterion)
    layerOutputs = torch.zeros(trainInput:size(1), modules[i].outputs)
    local tempstate = utils.deepCopyTable(state)
    totalLoss = 0

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

    for k = 1, epochs do
      local shuffle = torch.randperm(trainInput:size(1)):long()

      verbose('Epoch ' .. k .. ' of ' .. epochs)

      for j = 1, trainInput:size(1), batchSize do
        if (j-1) % 100 == 0 then
          verbose( 'Example ' .. j .. ' of ' .. trainInput:size(1)
                .. ', avgLoss: ' .. totalLoss / j)
        end
        input = trainInput:index(1, shuffle[{{j, math.min(j + batchSize - 1, trainInput:size(1))}}])
        target = input

        method(feval, params, tempstate)
        collectgarbage()
      end
    end

    if profile then
      ProFi:checkMemory(nil, 'Before creating input for next layer (' .. i .. ')')
    end

    layerOutputs = encoder:forward(trainInput)

--    if i < #modules then
      trainInput = layerOutputs:clone()
  --  end

    if profile then
      ProFi:checkMemory(nil, 'After making input, before stacking (' .. i .. ')')
    end

    encoder:clearState()
    stackedAutoencoder:add(encoder)

    if profile then
      ProFi:checkMemory(nil, 'After stacking (' .. i .. ')')
    end
  end

  if optimize then
    local mem1 = optnet.countUsedMemory(stackedAutoencoder)
    optnet.optimizeMemory(stackedAutoencoder, trainset[1][1], optnetOptions)
    local mem2 = optnet.countUsedMemory(stackedAutoencoder)
    verbose('Overall: ' .. mem1.total_size/mega .. ' MBytes => '
          .. mem2.total_size/mega .. ' MBytes')
  end

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

    local params, feval
    if finetune == 'all' then
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

    for j = 1, epochs do
      totalLoss = 0
      local shuffle = torch.randperm(trainInput:size(1)):long()
      verbose('Epoch ' .. j .. ' of ' .. epochs)
      for i = 1, trainInput:size(1), batchSize do
        if (i-1) % 100 == 0 then
          verbose( 'Example ' .. i .. ' of ' .. trainInput:size(1)
                .. ', avgLoss: ' .. totalLoss / i)
        end
        input = trainInput:index(1, shuffle[{{i, math.min(i + batchSize, trainInput:size(1))}}])
        target = trainTarget:index(1, shuffle[{{i, math.min(i + batchSize, trainInput:size(1))}}])

        method(feval, params, tempstate)
        totalLoss = totalLoss + loss
      end
    end
  end

  print(stackedAutoencoder:forward(trainInput))

  -- Clear internal state (temporary variables) of the network
  -- Makes the network much smaller in memory
  stackedAutoencoder:clearState()

  return stackedAutoencoder
end

return gl
