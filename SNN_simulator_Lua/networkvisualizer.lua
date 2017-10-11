require 'nn'
require 'image'
require 'gnuplot'
local utils = require 'utils'

local network = torch.load(arg[1], 'ascii')

local images = {}

local samples = 10

for i = 2, #arg do
  local path, file, ext = utils.parsePath(arg[i])

  if ext == 'th7' then
    local data = torch.load(arg[i], 'ascii')
    for j = 1, samples do
      local sample = math.random(data[1]:size(1))
      images['[' .. file .. '][' .. sample .. ']'] = data[1][sample]
    end
  else
    images['[' .. file .. ']'] = image.load(arg[i])
  end
end

local function forEachModule(f, network, name)
  if network.modules then
    for key, module in pairs(network.modules) do
      forEachModule(f, module, name .. type(network) .. '[' .. key .. '].')
    end
  else
    f(network, name .. type(network))
  end
end

local function getResponses(network, images)
  local weights = {}
  local responses = {}

  local function save_weight(module, name)
    if module.weight then
      weights[name] = module.weight
    end
  end

  local function save_response(imagename)
    responses[imagename] = {}
    return function(module, name)
      if module.output and type(module.output) ~= 'table' then
        responses[imagename][name] = module.output:clone()
      end
    end
  end

  forEachModule(save_weight, network, '')

  for imgName, img in pairs(images) do
    network:forward(img)
    forEachModule(save_response(imgName), network, '')
  end

  network:clearState()

  return weights, responses
end

local weights, responses = getResponses(network, images)

local function makeDir(dir)
  if not paths.dir(dir) then
    paths.mkdir(dir)
  end
end

makeDir('visualizations')

local visDir = paths.concat('visualizations', arg[1])

makeDir(visDir)

local filDir = paths.concat(visDir, 'filters')
local resDir = paths.concat(visDir, 'responses')

makeDir(filDir)
makeDir(resDir)

local function visMat(mat, dir)
  local function save(mat, name)
      gnuplot.pngfigure(paths.concat(dir, name))
      gnuplot.imagesc(mat)
      gnuplot.plotflush()    
  end
--print(mat)
print('Saving a matrix of size ' .. tostring(mat:size()))
  if mat:dim() > 3 then
    for i = 1, mat:size(1) do
      local subDir = paths.concat(dir, tostring(i))
      makeDir(subDir)
      visMat(mat[i], subDir)
    end
  elseif mat:dim() == 3 then
    for i = 1, mat:size(1) do
      save(mat[i], tostring(i))
    end
  elseif mat:dim() == 2 then
    save(mat, '1')
  else
    local w, h = utils.findDims(mat:nElement())
    save(mat:reshape(w, h), '1')
  end
end

for k, v in pairs(weights) do
  local vDir = paths.concat(filDir, k)
  makeDir(vDir)
  visMat(v, vDir)
end

for k, v in pairs(responses) do
  local vDir = paths.concat(resDir, k)
  makeDir(vDir)
  for k2, v2 in pairs(v) do
    local v2Dir = paths.concat(vDir, k2)
    makeDir(v2Dir)
    visMat(v2, v2Dir)
  end
end
