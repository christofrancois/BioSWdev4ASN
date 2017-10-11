require 'gnuplot'
require 'nn'
require 'sortlayer'

-- randomly pick a picture from a dataset and plot it as grayscale

cmd = torch.CmdLine()
cmd:text()
cmd:text('Using neural networks over the network')
cmd:text('Options:')
cmd:option('--dataset', '', 'which dataset to use. A lua table of samples.')
cmd:option('--network', '', 'giving a network will draw the responses of its conv layers.')
cmd:text()
opt = cmd:parse(arg or {})

math.randomseed(os.time())

local function outputImage(module)
  local pic = module.output[1]

  if pic:dim() == 4 then
    pic = pic:reshape(pic:size(1), pic:size(2), pic:size(3))
  end

  if pic:dim() == 3 then

    for i = 1, pic:size(1) do
      gnuplot.figure()

      gnuplot.imagesc(pic[i])
    end
  end
end

function forAllModules(module, f, level)
  level = level or 0

  if module.modules then
    for _, v in pairs(module.modules) do
      forAllModules(v, f, level + 1)
    end
  else
    f(module, level)
  end
end

function mat2img(mat)
  if mat:dim() > 2 then
    for i = 1, mat:size(1) do
      mat2img(mat[i])
    end
  elseif mat:dim() == 2 then
    gnuplot.figure()
    gnuplot.imagesc(mat)
  end
end

if opt.dataset ~= '' then
  local augment = require 'augment'
  local set = augment.normalize(torch.load(opt.dataset, 'ascii'))
  local net = opt.network ~= '' and torch.load(opt.network, 'ascii') or nil

  local pic = set[1][math.random(set[1]:size(1))]

  local mean, std = torch.mean(pic), torch.std(pic)
  print(mean, std)
  print(pic:size())

--[[
gnuplot.imagesc(0.2126 * pic[1] + 0.7152 * pic[2] + 0.0722 * pic[3])
pic = pic:reshape(1,pic:size(1),pic:size(2),pic:size(3))
--set2 = augment.jitter({pic,torch.zeros(1)},1,5,1)
--set2 = augment.saltAndPepper({pic,torch.zeros(1)},0.1,1,false)
set2 = augment.zoom({pic,torch.zeros(1)},0.5,1,false)
pic = set2[1][set2[1]:size(1)]
gnuplot.figure()]]
  if pic:size(1) == 1 then
    gnuplot.imagesc(pic[1])
  else
    gnuplot.imagesc(0.2126 * pic[1] + 0.7152 * pic[2] + 0.0722 * pic[3])
  end

  if net then
    net:evaluate()
    net:forward(pic:reshape(1, pic:size(1), pic:size(2), pic:size(3)):repeatTensor(2,1,1,1))
    net:applyToModules(outputImage)
    forAllModules(net, function(module, level) if module.weight then print(string.rep(' ', level) .. (module.weight and torch.sum(torch.le(module.weight, torch.zeros(module.weight:size()))) or '')); --[[mat2img(module.weight)]] end end)
  end
end
