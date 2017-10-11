require 'nn'
require 'optim'

local modules = nn.Sequential()

--local module1 = nn.ConcatTable()
--module1:add(nn.Narrow(2, 1, 84))
--module1:add(nn.Narrow(2, 85, 2))
modules:add(nn.Narrow(2, 1, 84))

local classes = 4
--[[
local module2 = nn.ParallelTable()
module2:add(nn.Linear(84, classes))
module2:add(nn.Identity())

modules:add(module1)
modules:add(module2)
modules:add(nn.JoinTable(1, 1))
]]
--modules:add(nn.SelectTable(1))
modules:add(nn.Linear(84, classes))

return
  { modules =
    { { module = modules, inputs = 86, outputs = classes, criterion = nn.MSECriterion() }
    }
  , finetuneCrit = nn.CrossEntropyCriterion()
  , trainset = function(dataset)
--print(dataset[2][{{}, 2}])
      return { dataset[1], dataset[2][{{}, 2}] + 1 }
    end
--  , method = 'optim.sgd'
  , method = function(f, x, s)
      local config =
        { learningRate = 0.001
        , beta1 = 0.9
        , beta2 = 0.999
        , epsilon = 1e-8
        }
      optim.adam(f, x, config, s)
    end
  , epochs = 300
  , batchSize = 5
  , normalize = false
  }
