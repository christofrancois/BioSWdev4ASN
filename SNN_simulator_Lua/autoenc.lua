require 'dp'
require 'feedback/imagefeedback'

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
cmd:option('--hiddenSize', '{78}', 'number of hidden units per layer')
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
for i,hiddenSize in ipairs(opt.hiddenSize) do
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters
   if opt.batchNorm then
      model:add(nn.BatchNormalization(hiddenSize))
   end
   model:add(nn[opt.activation]())
   if opt.dropout then
      model:add(nn.Dropout())
   end
   inputSize = hiddenSize
end

-- output layer
model:add(nn.Linear(inputSize, originalSize))
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
   feedback = dp.ImageFeedback{width = 28, name = 'test', frequency = 20},
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

print(xp:optimizer())
