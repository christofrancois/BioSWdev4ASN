require 'nn'
require 'optim'
require 'LargeMarginCriterion'

local lfs = require 'lfs'

local resultDir = './results'

local datafile = 'configData.arff'

local numAcc = true

local attribs =
  { { name = 'method'
    , type = '{ sgd, adam }'
    , get = function(x) return x.config.config.desc.method.method end
    }
  , { name = 'lrate'
    , type = '{ 1e-01, 1e-02, 1e-03, 1e-04, 1e-05 }'
    , get = function(x) return string.format('%.e', x.config.config.desc.method.learningRate) end
    }
  , { name = 'lrdecay'
    , type = '{ false, true }'
    , get = function(x) return tostring(x.config.config.desc.lrateDecay) end
    }
  , { name = 'residual'
    , type = '{ false, true }'
    , get = function(x) return tostring(x.config.config.desc.residual) end
    }
  , { name = 'init'
    , type = '{ kaiming, lsuv }'
    , get = function(x) return x.config.config.desc.init end
    }
  , { name = 'dropout'
    , type = '{ false, true }'
    , get = function(x) return tostring(x.config.config.desc.dropout) end
    }
  , { name = 'criterion'
    , type = '{ lmdnn, cross-only }'
    , get = function(x) return x.config.config.desc.crit end
    }
  , { name = 'accuracy'
    , type = numAcc and 'NUMERIC' or '{ under90, over90 }'
    , get = function(x) return (numAcc and x.config.accuracy) or (x.config.accuracy >= 0.9 and 'over90' or 'under90') end
    }
  }

local header = [[
@RELATION dnnconfig

]]

for _, att in pairs(attribs) do
  header = header .. '\n@ATTRIBUTE ' .. att.name .. ' ' .. att.type
end

local datatext = '\n\n@DATA\n'

for file in lfs.dir(resultDir) do
  if file ~= '.' and file ~= '..' then
    print(file)
    local path = resultDir .. '/' .. file
    local content = torch.load(path, 'ascii')

--[[
    content.config.config.finetuneCrit.output = nil
    content.config.config.finetuneCrit.gradInput = nil
    for _, v in pairs(content.config.config.modules) do
      v.module:clearState()
    end

    torch.save(path, content, 'ascii')
]]

    local first = true

    for _, att in pairs(attribs) do
      if first then
        first = false
      else
        datatext = datatext .. ','
      end
      datatext = datatext .. att.get(content)
    end

    datatext = datatext .. '\n'
  end
end

local file = assert(io.open(datafile, 'w'))
file:write(header .. datatext)
file:close()
