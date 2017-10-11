require 'nn'
require 'optim'

return
  { modules =
    { { module = nn.Linear(2, 1), inputs = 2, outputs = 1 }
    }
  , trainset =
    { torch.DoubleTensor({{0, 0}, {0, 1}, {1, 0}, {1, 1}})
    , torch.DoubleTensor({0, 1, 1, 0})
    }
  , finetuneCrit = nn.MSECriterion()
  , method = 'optim.sgd'
  , epochs = 100
  , batchSize = 1
  }
