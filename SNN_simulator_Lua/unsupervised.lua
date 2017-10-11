--[[require 'nn'
require 'gnuplot'

local mnist_train = torch.load('/home/lehtolav/data/mnist/train.th7', 'ascii')

local model = nn.Sequential()

local hiddenSize = 50
local dataSize = 28*28
local transfer = nn.Sigmoid()

model:add(nn.Linear(dataSize, hiddenSize))
model:add(transfer)
model:add(nn.Linear(hiddenSize, dataSize))

local datasize = 60000
local dataset = {}

function dataset:size() return datasize end

for i = 1, dataset:size() do
  local data = mnist_train[1][i]:reshape(28*28):double()
  dataset[i] = {data, data}
end

local activation = torch.zeros(hiddenSize)

for i = 1, datasize do
  model:forward(mnist_train[1][i]:reshape(28*28):double())
  activation:add(transfer.output)
end

activation:div(datasize)

local criterion = nn.MSECriterion()
local trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.1
trainer.learningRateDecay = 0.001
trainer.maxIteration = 10
trainer:train(dataset)]]

require 'dp'
require 'feedback/imagefeedback'
require 'unsup'
local aeviz = require 'aevisualizer'

local oldzca = dp.ZCA.fit
dp.ZCA.fit = function(self, X)
  oldzca(self, X:double())
  self._P = self._P:float()
  self._mean = self._mean:float()
end

--[[command line arguments]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Classification using MLP Training/Optimization')
cmd:text('Example:')
cmd:text('$> th neuralnetwork.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--lrDecay', 'linear', 'type of learning rate decay : adaptive | linear | schedule | none')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 300, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--schedule', '{}', 'learning rate schedule')
cmd:option('--maxWait', 4, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.001, 'factor by which learning rate is decayed for adaptive decay.')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--activation', 'ReLU', 'transfer function like ReLU, Tanh, Sigmoid')
cmd:option('--hiddenSize', '78', 'number of hidden units per layer')
cmd:option('--batchSize', 32, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 20, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dropout', false, 'apply dropout on hidden neurons')
cmd:option('--batchNorm', false, 'use batch normalization. dropout is mostly redundant with this')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--lecunlcn', false, 'apply Yann LeCun Local Contrast Normalization')
cmd:option('--progress', false, 'display progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})
opt.schedule = dp.returnString(opt.schedule)
opt.hiddenSize = dp.returnString(opt.hiddenSize)
if not opt.silent then
   table.print(opt)
end

--[[preprocessing]]--

local input_preprocess = {}
if opt.standardize then
   table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
   table.insert(input_preprocess, dp.ZCA())
end
if opt.lecunlcn then
   table.insert(input_preprocess, dp.GCN())
   table.insert(input_preprocess, dp.LeCunLCN{progress=true})
end

--[[data]]--

if opt.dataset == 'Mnist' then
   ds = dp.Mnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'NotMnist' then
   ds = dp.NotMnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar10' then
   ds = dp.Cifar10{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar100' then
   ds = dp.Cifar100{input_preprocess = input_preprocess}
elseif opt.dataset == 'FaceDetection' then
   ds = dp.FaceDetection{input_preprocess = input_preprocess}
else
   error("Unknown Dataset")
end

--pp = ConstantPP(ds:trainSet():inputs())
--ds:trainSet():targets():forward('b'):replace('bf', ds:trainSet():inputs())
--ds:validSet():targets():forward('b'):replace('bf', ds:validSet():inputs())
--ds:testSet():targets():forward('b'):replace('bf', ds:testSet():inputs())

local trainInput = dp.DataView('bf', ds:trainSet():inputs():forward('bf'))
local train = dp.DataSet{ inputs = ds:trainSet():inputs(), targets = trainInput, which_set = 'train'}
local validInput = dp.DataView('bf', ds:validSet():inputs():forward('bf'))
local valid = dp.DataSet{ inputs = ds:validSet():inputs(), targets = validInput, which_set = 'valid'}
local testInput = dp.DataView('bf', ds:testSet():inputs():forward('bf'))
local test = dp.DataSet{ inputs = ds:testSet():inputs(), targets = testInput, which_set = 'test'}

ds = dp.DataSource{ train_set = train, valid_set = valid, test_set = test}

--[[Model]]--

model = nn.Sequential()
model:add(nn.Convert(ds:ioShapes(), 'bf')) -- to batchSize x nFeature (also type converts)

-- hidden layers
originalSize = ds:featureSize()
inputSize = originalSize
hiddenSize = opt.hiddenSize
local hiddenLayer = nn.Linear(inputSize, hiddenSize)
model:add(hiddenLayer) -- parameters
local transfer = nn[opt.activation]() -- save the transfer layer for later data extraction
model:add(transfer)
inputSize = hiddenSize

-- output layer
local outputLayer = nn.Linear(inputSize, originalSize)
model:add(outputLayer)
--model:add(nn.LogSoftMax())


--[[Propagators]]--
if opt.lrDecay == 'adaptive' then
   ad = dp.AdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}
elseif opt.lrDecay == 'linear' then
   opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
end

train = dp.Optimizer{
   acc_update = opt.accUpdate,
   loss = nn.ModuleCriterion(nn.MSECriterion(), nil, nn.Convert()),
   epoch_callback = function(model, report) -- called every epoch
      -- learning rate decay
      if report.epoch > 0 then
         if opt.lrDecay == 'adaptive' then
            opt.learningRate = opt.learningRate*ad.decay
            ad.decay = 1
         elseif opt.lrDecay == 'schedule' and opt.schedule[report.epoch] then
            opt.learningRate = opt.schedule[report.epoch]
         elseif opt.lrDecay == 'linear' then 
            opt.learningRate = opt.learningRate + opt.decayFactor
         end
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
         end
      end
   end,
   callback = function(model, report) -- called for every batch
      if opt.accUpdate then
         model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
      else
         model:updateGradParameters(opt.momentum) -- affects gradParams
         model:updateParameters(opt.learningRate) -- affects params
      end
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   feedback = dp.Feedback{name = 'trainer'},--dp.ImageFeedback{width = 28, name = 'trainer', frequency = 20},
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   progress = opt.progress
}
valid = dp.Evaluator{
   feedback = dp.Feedback{name = 'eval'},
   sampler = dp.Sampler{batch_size = opt.batchSize}
}
test = dp.Evaluator{
   feedback = dp.Feedback{name = 'test'},
   sampler = dp.Sampler{batch_size = opt.batchSize}
}

--[[Experiment]]--

xp = dp.Experiment{
   model = model,
   optimizer = train,
   validator = valid,
   tester = test,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'random_seed'},--{'validator','loss'},
         maximize = true,
         max_epochs = opt.maxTries
      },
      ad
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--

if opt.cuda then
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Model :"
   print(model)
end

xp:run(ds)

require 'gnuplot'

aeviz.visualize(hiddenLayer.weight, 28)
gnuplot.figure()
aeviz.visualize(outputLayer.weight:t(), 28)

-- Use training set again to find correlations between hidden layer
-- transfer function values and correct classes
-- dp library obfuscates the training data such that it cannot be used outside dp classes
local mnist_train = torch.load('/home/lehtolav/data/mnist/train.th7', 'ascii')
local classes = {}
local classN = {}
local datasize = 60000
local predictors = 15 -- number of predictors per class
for i = 0, 9 do
  classes[i] = torch.zeros(hiddenSize)
  classN[i] = 0
end

local function normalize(vector)
  return torch.div(vector, torch.sum(vector))
end

for i = 1, datasize do
  model:forward(mnist_train[1][i]:reshape(28*28):double())
  --local _, topk = torch.topk(transfer.output, predictors, 1, true)
  --topk:apply(
  --  function(x) classes[mnist_train[2][i]][x] = classes[mnist_train[2][i]][x] + 1 end
  --)
  classes[mnist_train[2][i]]:add(transfer.output)
  classN[mnist_train[2][i]] = classN[mnist_train[2][i]] + 1
end

for i = 0, 9 do
  --_, classes[i] = torch.topk(classes[i], predictors, 1, true)
  classes[i] = normalize(classes[i]:div(classN[i]))
--print(classes[i])
end

-- Use test set to test classifying accuracy
local mnist_test = torch.load('/home/lehtolav/data/mnist/test.th7', 'ascii')
local testSize = 10000
local correct = 0

for i = 1, testSize do
  model:forward(mnist_test[1][i]:reshape(28*28):double())
  --gnuplot.imagesc(torch.reshape(model.output, 28, 28))
  --local norm_out = torch.div(transfer.output, torch.norm(transfer.output))
  local corr_class = mnist_test[2][i]
  local best, best_ind = nil, nil
  for j = 0, 9 do
    local err = torch.dist(classes[j], normalize(transfer.output))--transfer.output:index(1, classes[j]):sum() --torch.dist(norm_out, classes[j])
    --print(err, best)
    if best == nil or err < best then
      best, best_ind = err, j
    end
  end

  print('predicted ' .. best_ind .. ', correct was ' .. corr_class)
  print('error was ' .. best)

  if best_ind == corr_class then
    correct = correct + 1
  end
end

print(correct .. ' correct out of ' .. testSize .. ' (' .. 100*correct/testSize .. '%)')
