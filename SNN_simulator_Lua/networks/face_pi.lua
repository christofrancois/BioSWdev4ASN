require 'nn'
require 'optim'

local activation = nn.ReLU
local linear = nn.Linear

local channels = 3
local w1 = 100
local h1 = 100

local conv1 = nn.Sequential()
local fi1, fo1, ks1 = 3, 6, 5
conv1:add(nn.View(channels, w1, h1))
conv1:add(renorm(nn.SpatialConvolution(fi1, fo1, ks1, ks1)))
conv1:add(activation())

local conv2 = nn.Sequential()
local fi2, fo2, ks2 = fo1, 16, 5
conv2:add(nn.SpatialMaxPooling(2,2,2,2))
local w2, h2 = (w1 - ks1 + 1)^2 / 2, (h1 - ks1 + 1)^2 / 2
conv2:add(renorm(nn.SpatialConvolution(fi2, fo2, ks2, ks2)))
conv2:add(activation())

local lin1 = nn.Sequential()
lin1:add(nn.SpatialMaxPooling(2,2,2,2))
local w3, h3 = (w2 - ks2 + 1)^2 / 2, (h2 - ks2 + 1)^2 / 2
lin1:add(nn.View(fo2 * w3 * h3))
lin1:add(renorm(linear(fo2 * w3 * h3, 120)))
lin1:add(activation())

local lin2 = nn.Sequential()
lin2:add(renorm(linear(120, 84)))
lin2:add(activation())

local lin3 = renorm(linear(84, 1))

local modules =
  { { module = conv1
    , inputs = picw*pich
    , outputs = fo1 * w2 * h2 * 4 
    , criterion = nn.MSECriterion()
    }
  , { module = conv2
    , inputs = fo1 * w2 * h2 * 4
    , outputs = fo2 * w3 * h3 * 4
    , criterion = nn.MSECriterion()
    }
  , { module = lin1
    , inputs = fo2 * w3 * h3
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
    , outputs = 1
    , criterion = nn.MSECriterion()
    }
  }

return
  { modules = modules
  , trainset = {}
  , finetuneCrit = nn.MSECriterion()
  , method = 'optim.sgd'
  , epochs = 100
  , batchSize = 1
  }
