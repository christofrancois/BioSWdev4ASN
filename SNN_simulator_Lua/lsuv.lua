-- LSUV method for initialization
-- Works for (some) container modules as well as for the custom tables used in the greedylayerwise method

local function getClassName(module)
  return tostring(module):match('^[^ ]*')
end

--[[
   /--
--<
   \--
]]
local function isMultiOutput(module)
  --TODO: add more cases
  --(for now only cases relevant to current study)
  if getClassName(module) == 'nn.ConcatTable' then
    return true
  end
  return false
end

--[[
--\
   >--
--/
]]
local function isMultiInput()
  --TODO: same
  if getClassName(module) == 'nn.CAddTable' then
    return true
  end
  return false
end

return function(config)

  local args, network, batch, tolerance, maxTrials = xlua.unpack(
      {config},
      'greedyLayerwise', 
      'Greedy layer-wise training for deep networks',
      {arg = 'network', type = 'table',
       help = 'Array of modules and their input and output dimensions / plain network'},
      {arg = 'batch', type = 'table',
       help = 'Training data batch to base the initialization on'},
      {arg = 'tolerance', type = 'number', default = 0.01,
       help = 'Tolerance for variance'},
      {arg = 'maxTrials', type = 'number', default = 100,
       help = 'Maximum number of iterations to run per layer'}
  )

  if not network[1] then -- not custom structure for greedylayerwise
      -- standardize input to the custom input style
      network = { modules = network }
  end

  local function handleModules(module, input)
--print('into "' .. getClassName(module) .. '"')
    if getClassName(module) == 'nn.SpatialBatchNormalization'
    or getClassName(module) == 'nn.BatchNormalization' then
      -- Not sure how to handle these
      -- Letting the procedure run causes a 'misaligned parameter' error(?)
      --return input --module:forward(input)
    elseif module.weight then
      module.weight = torch.randn(module.weight:size())

      if module.weight:dim() > 2 then
        -- Conv layer
        for i = 1, module.weight:size(1) do
          for j = 1, module.weight:size(2) do
            local _, _, V = torch.svd(module.weight[i][j])
            module.weight[i][j] = V:t()
          end
        end
      elseif module.weight:dim() == 2 then
        local _, _, V = torch.svd(module.weight)
        module.weight = V:t()
      end -- and if dim == 1 we do nothing

      for i = 1, maxTrials do
        module:forward(input)
        local var = torch.var(module.output)

        if math.abs(var - 1) < tolerance then
          break
        end

        module.weight = module.weight / math.sqrt(var)
      end

      module.weight = module.weight:contiguous()

      module:forward(input)
      input = module.output:clone()
      module:clearState()
    elseif module.modules then
      local multiIn = isMultiInput(module)
      local multiOut = isMultiOutput(module)

      if multiIn then
        -- input should be a table
        local output = {}
        for i = 1, #module.modules do
          output[i] = handleModules(module.modules[i], input[i])
        end
        if multiOut then
          -- e.g. nn.ParallelTable
          input = output
        else
          module:forward(input)
          input = module.output:clone()
          module:clearState()
        end
      elseif multiOut then
        local output = {}
        for i = 1, #module.modules do
          output[i] = handleModules(module.modules[i], input)
        end
        input = output
      else
        for i = 1, #module.modules do --break
          input = handleModules(module.modules[i], input)
        end
      end
    else
      module:forward(input)
      input = module.output:clone()
      module:clearState()
    end

--print('out of "' .. getClassName(module) .. '"')

    return input
  end

  local input = batch

  for i = 1, #network do
    print('Handling module ' .. i .. ' / ' .. #network)
--    print(network[i].inputs, network[i].outputs)
    input = handleModules(network[i].module, input)
--    print(input:size())
  end

  if #network == 1 and not network[1].inputs then
    return network[1]
  end

  return network
end

