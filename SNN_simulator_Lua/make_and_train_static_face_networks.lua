-- Restructures the trained larvi network to output the face detection bit
-- parallel to its generated features. Uses the created network to generate
-- a training dataset for face detection from the features.
-- As a bonus feature, prints the accuracy of face detection.

require 'nn'
require 'sortlayer'
local utils = require 'utils'
local lfs = require 'lfs'
local augment = require 'augment'

local function initSpatial(layer)
  layer:init('weight', nninit.eye)
       :init('weight', nninit.mulConstant, 1/2)
       :init('weight', nninit.addNormal, 0, 0.01)
       :init(getBias, nninit.constant, 0)
  return layer
end

local function initLinear(layer)
  layer:init('weight', nninit.kaiming, { gain = 'relu' })
       :init(getBias, nninit.constant, 0)
  return layer
end

local function exists(filename)
  local fsize = lfs.attributes(filename, 'size')
  return (fsize and fsize > 0) or false
end

--local validator = utils.nClassValidator

local networkSettings = '' -- '--address 169.254.5.208 --port 4983 --localaddress 169.254.197.218'

--local prefix = 'staticFace'
--local postfix = '224'
local prefix = 'fkd'
local postfix = '96'
local folder = '/worktmp/' .. prefix .. '_' .. postfix
local testname = folder .. '/' .. prefix .. 'Test' .. postfix .. '.th7'
local trainname = folder .. '/' .. prefix .. 'Train' .. postfix .. '.th7'
local augname = folder .. '/' .. prefix .. 'TrainAug' .. postfix .. '.th7'
local testaugname = folder .. '/' .. prefix .. 'TestAug' .. postfix .. '.th7'
local featname = folder .. '/' .. prefix .. 'TrainFeat' .. postfix .. '.th7'

local augdefname = folder .. '/' .. prefix .. '_augment_' .. postfix .. '.lua'
local compdefname = folder .. '/' .. prefix .. '_complete_' .. postfix .. '.lua'
local discrdefname = folder .. '/' .. prefix .. '_discriminator_' .. postfix .. '.lua'
local recogdefname = folder .. '/' .. prefix .. '_recognizer_' .. postfix .. '.lua'

local compnetname = folder .. '/' .. prefix .. '_complete_' .. postfix .. '.net'
local discrnetname = folder .. '/' .. prefix .. '_discr_' .. postfix .. '.net'
local recognetname = folder .. '/' .. prefix .. '_recog_' .. postfix .. '.net'
local finalnetname = folder .. '/' .. prefix .. '_combined_' .. postfix .. '.net'

if not exists(augname) then
  print('Augmenting training data.')
  os.execute('th trainingserver.lua ' .. networkSettings .. ' --network ' .. augdefname .. ' --dataset ' .. trainname .. ' --output ' .. augname)
end

if not exists(testaugname) then
  print('Augmenting testing data.')
  os.execute('th trainingserver.lua ' .. networkSettings .. ' --network ' .. augdefname .. ' --dataset ' .. testname .. ' --output ' .. testaugname)
end

if not exists(compnetname) then
  print('Training complete network.')
  os.execute('th trainingserver.lua ' .. networkSettings .. ' --network ' .. compdefname .. ' --dataset ' .. augname .. ' --output ' .. compnetname)
end

print('Loading complete network')

local larviNet = torch.load(compnetname, 'ascii')

local testset = torch.load(testaugname, 'ascii')

--local mean, std = torch.mean(trainset[1]), torch.std(trainset[1])

local mean, std = torch.mean(testset[1]), torch.std(testset[1])
testset[1] = (testset[1] - mean) / std

local accuracy = utils.nClassValidator(larviNet, {testset[1], testset[2]:double() + 1})

print('Test accuracy ' .. accuracy * 100 .. '%')

larviNet:remove()
larviNet:remove()

local trainset

-- No need to load or testing, trainingserver loads it itself, feature generation loads different set
--if not exists(augname) or not exists(featname) then
--  print('Loading training data')
--  trainset = torch.load(trainname, 'ascii')
--end

--local trainset = torch.load('larviTrainAugFeat.th7', 'ascii')

if not exists(featname) then
  print('Generating features.')
  trainset = torch.load(augname, 'ascii')

  local nData = trainset[2]:nElement()
  local feats = torch.zeros(nData, 120)

  for i = 1, nData, 200 do
    print(i .. '/' .. nData)
    larviNet:forward(trainset[1][{{i, math.min(i + 200, nData)}}])--trainset[1]:narrow(1, i, math.min(i + 200, nData) - i))
    --local selection = feats:narrow(1, i, math.min(i + 200, nData) - i)
    --selection = larviNet.output
    feats[{{i, math.min(i + 200, nData)}}] = larviNet.output
  end

  torch.save(featname, { feats, trainset[2] }, 'ascii')
  larviNet:clearState()
end

if not exists(discrnetname) then
  print('Training discriminator.')
  os.execute('th trainingserver.lua ' .. networkSettings .. ' --network ' .. discrdefname .. ' --dataset ' .. featname .. ' --output ' .. discrnetname)
end

if not exists(recognetname) then
  print('Training recognizer.')
  os.execute('th trainingserver.lua ' .. networkSettings .. ' --network ' .. recogdefname .. ' --dataset ' .. featname .. ' --output ' .. recognetname)
end

print('Loading discriminator')
local larviDiscr = torch.load(discrnetname, 'ascii')

print('Loading recognizer')
local larviRecog = torch.load(recognetname, 'ascii')

larviNet:forward(testset[1])
local testfeat = larviNet.output
local trainfeat = torch.load(featname, 'ascii')

accuracy = utils.hingeValidator(larviDiscr, augment.hingeTargets(trainfeat))
print('Training accuracy of discriminator ' .. accuracy * 100 .. '%')

accuracy = utils.hingeValidator(larviDiscr, augment.hingeTargets({testfeat, testset[2]:double()}))
print('Test accuracy of discriminator ' .. accuracy * 100 .. '%')

local function onlyLarvi(dataset)
  local newset = { }

  local entries = dataset[2]:size(1) - torch.sum(torch.eq(torch.zeros(dataset[2]:size()), dataset[2]))

  newset[1] = torch.zeros(entries, dataset[1]:size(2))
  newset[2] = torch.zeros(entries)
  local current = 1

  for i = 1, dataset[2]:size(1) do
    if dataset[2][i] ~= 0 then
      newset[1][current] = dataset[1][i]
      newset[2][current] = dataset[2][i]
      current = current + 1
    end
  end

  return newset
end

--accuracy = utils.nClassValidator(larviRecog, onlyLarvi({trainfeat[1], trainfeat[2]:double()}))
print('Training accuracy of recognizer ' .. accuracy * 100 .. '%')

accuracy = utils.nClassValidator(larviRecog, onlyLarvi({testfeat, testset[2]:double()}))
print('Test accuracy of recognizer ' .. accuracy * 100 .. '%')

larviNet:clearState()

local newLayer = nn.ConcatTable()
newLayer:add(larviDiscr)
newLayer:add(nn.Identity())

newLayer:clearState()

larviNet:add(newLayer)
larviNet:add(nn.JoinTable(1, 1))

larviNet:clearState()

torch.save(finalnetname, larviNet, 'ascii')
