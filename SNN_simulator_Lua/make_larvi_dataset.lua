-- Collects the larvi dataset pictures, preprocesses them if required
-- and outputs two torch-compatible files containing the dataset
-- divided to training and test data.

require 'image'
require 'lfs'
local resmets = require 'resamplers'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Collects the larvi dataset pictures, preprocesses them if required')
cmd:text('and outputs a single torch-compatible file containing the dataset')
cmd:text('divided to training and test data.')
cmd:text()
cmd:text('Options:')
cmd:option('--w', 0, 'width of results')
cmd:option('--h', 0, 'height of results')
cmd:option('--tw', 0, 'width of training images (omit to use w)')
cmd:option('--th', 0, 'height of training image (omit to use h)')
cmd:option('--ntest', 10, 'number of test samples per label')
cmd:option('--resampler', 'nn', 'resampling method')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})
local verbose
if opt.silent then
  verbose = function() end
else
  --table.print(opt)
  verbose = print
end

local testN = opt.ntest

local width = opt.w
local height = opt.h

if width == 0 or height == 0 then
  error('You must define the width and height')
end

trainWidth = opt.tw == 0 and width or opt.tw
trainHeight = opt.th == 0 and height or opt.th

-- 'nn' for nearest neighbor
-- 'bl' for bilinear
-- 'mp' for maxpooling
local resampler = opt.resampler

local larvidirs =
  { './datapics/nolarvi/'
  , './datapics/larvi/'
  , './datapics/francois/'
  , './datapics/timo/'
  , './datapics/mikko/'
  }

local dataset = {}
dataset[1] = {}
dataset[2] = {}
dataset[1][1] = {}
dataset[1][2] = {}
dataset[2][1] = {}
dataset[2][2] = {}

local count = 1
local labelCount = 1

for picLabel, picDir in pairs(larvidirs) do
  labelCount = 1

  for picFile in lfs.dir(picDir) do
    local namelen = picFile:len()
    --print(picFile)

    if picFile:sub(namelen - 3, namelen) == '.jpg' then
      verbose('Processing image ' .. count)
      local pic = image.load(picDir .. picFile)
      --print(pic:size())
      if labelCount <= testN then
        pic = resmets[resampler](pic, width, height)
        dataset[2][1][#dataset[2][1] + 1] = pic
        dataset[2][2][#dataset[2][2] + 1] = picLabel - 1
      else
        pic = resmets[resampler](pic, trainWidth, trainHeight)
        dataset[1][1][#dataset[1][1] + 1] = pic
        dataset[1][2][#dataset[1][2] + 1] = picLabel - 1
      end

      count = count + 1
      labelCount = labelCount + 1
    end
  end
end

local trainingSet = {}
local testingSet = {}
local channels = dataset[1][1][1]:size(1)
trainingSet[1] = torch.zeros(#dataset[1][1], channels, trainHeight, trainWidth)
trainingSet[2] = torch.zeros(#dataset[1][1])
testingSet[1] = torch.zeros(#dataset[2][1], channels, height, width)
testingSet[2] = torch.zeros(#dataset[2][1])

for i = 1, trainingSet[1]:size(1) do
  trainingSet[1][i] = dataset[1][1][i]
  trainingSet[2][i] = dataset[1][2][i]
end

for i = 1, testingSet[1]:size(1) do
  testingSet[1][i] = dataset[2][1][i]
  testingSet[2][i] = dataset[2][2][i]
end

torch.save('larviTrain.th7', trainingSet, 'ascii')
torch.save('larviTest.th7', testingSet, 'ascii')
