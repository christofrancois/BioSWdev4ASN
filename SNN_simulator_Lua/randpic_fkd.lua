require 'gnuplot'
require 'nn'
require 'sortlayer'
require 'image'

-- randomly pick a picture from a dataset and plot it as grayscale

cmd = torch.CmdLine()
cmd:text()
cmd:text('Using neural networks over the network')
cmd:text('Options:')
cmd:option('--dataset', '', 'which dataset to use. A lua table of samples.')
cmd:option('--network', '', 'giving a network will draw the responses of its conv layers.')
cmd:option('--test', false, 'testing mode: no targets are known')
cmd:text()
opt = cmd:parse(arg or {})

local testmode = opt.test

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

  local i = math.random(set[1]:size(1))
--i = 1118
  local pic = set[1][i]

  local mean, std = torch.mean(pic), torch.std(pic)
  --pic = (pic - mean) / std
  --pic = image.lcn(pic)
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
local cropw, croph = math.floor(17 / 2), math.floor(17 / 2)
local resw, resh = 11, 11

local function getCrop(pic, x, y)
  return image.scale(image.crop(pic, x - cropw, y - croph, x + cropw + 1, y + croph + 1), resw, resh)
end

--    pic[1] = image.lcn(pic[1])
--gnuplot.imagesc(pic[1])
--gnuplot.figure()
  gnuplot.imagesc(pic[1])

if not testmode then
  local x, y = set[2][i][1] + 1, set[2][i][2] + 1
  gnuplot.figure()
  --if pic:size(1) == 1 then
  local old = pic[1][y][x]
    pic[1][y][x] = 10
    gnuplot.imagesc(pic[1])
  pic[1][y][x] = old

gnuplot.figure()
gnuplot.imagesc(getCrop(pic[1], set[2][i][1], set[2][i][2]))
end

net:evaluate()

print('begin using network')
--nn.BatchNormalization.updateOutput = function(self, input)
--print(self.running_mean, self.running_var)
--  return (input - torch.mean(self.running_mean)) / math.pow(torch.mean(self.running_var), 0.5)
--end
--net:get(1):remove(1)
--net:get(net:size()):remove()
--net:forward(net:forward(pic[1]))--:repeatTensor(2, 1)))
--for j = 1, 15 do
--  map[j] = out[1][j]-- + out[1][{1,15}] --out[2] - out[1]
--end

  local xlimit = pic:size(3) - 2 * cropw - 1
  local ylimit = pic:size(2) - 2 * croph - 1

  local map = torch.zeros(15, ylimit, xlimit)
  local crops = torch.zeros(xlimit * ylimit, resw * resh)

  for x = 1, xlimit do
    for y = 1, ylimit do
--print(x,y)
      local crop = getCrop(pic[1], x+cropw, y+croph):double()
  --print(crop)
--  local out = net:forward(torch.cat(crop:reshape(resw * resh), torch.DoubleTensor({x/96,y/96}), 1))
      crops[x + (y - 1) * xlimit] = crop
    end
  end


      local out = net:forward(crops)--crop:reshape(1, resw * resh):repeatTensor(2, 1))
print(crops:size())
print(out:size())
      for j = 1, 15 do
        map[j] = out[{{}, j}]:reshape(ylimit, xlimit)--out[1][j]-- + out[1][{1,15}] --out[2] - out[1]
      end
--[[      for j = 1, 15 do
        map[j][y][x] = out[1][j]-- + out[1][{1,15}] --out[2] - out[1]
      end
    end
  end
]]

print('done using network')
--  local gmin = torch.min(map)
--  map = map + gmin
  local cmap = map:clone()
  local temppic = pic[1]:clone()

  local function getMaxima(mat)
    local maxx, locy = torch.max(mat, 1)
    local maxy, locx = torch.max(maxx, 2)
    locx = locx:squeeze()
    return locx + cropw, locy[1][locx] + croph, maxy
  end

  for j = 1, 15 do
    --map[j] = 2 * cmap[j] - torch.sum(cmap, 1)
    gnuplot.figure()
    gnuplot.imagesc(map[j])
    local locx, locy, max = getMaxima(map[j])
    print(locy, locx, max:squeeze())
    temppic[locy][locx] = 10
  end

  gnuplot.figure()
  gnuplot.imagesc(temppic)

  temppic = pic[1]:clone()
  local priority = { 3, 4, 5, 6, 12, 13 }
  local xs, ys = {}, {}

  local function mapToPix(mat, f)
    for i = 1, mat:size(2) do
      for j = 1, mat:size(1) do
        mat[j][i] = - f(i, j) * mat[j][i]
      end
    end
  end

  for _, j in pairs(priority) do
  --  map[j] = 2 * cmap[j] - torch.sum(cmap, 1)
    local locx, locy, max = getMaxima(map[j])
    temppic[locy][locx] = 10
    xs[j] = locx
    ys[j] = locy
  end

  local function distance(x, y)
    return math.sqrt(x*x + y*y)
  end

  local function between(x1, y1, x2, y2)
    local mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    local md = distance(math.abs(x1 - mx), math.abs(y1 - my))
    print(md)
    return function(x, y)
      return (2 * md - distance(math.abs(x - mx), math.abs(y - my))) / md
    end
  end

  local function above(x1, y1, div)
    div = div or 100
    return function(x, y)
      local dy = y1 - y -- positive when y above y1
      local dx = math.abs(x - x1)
      local aboveness = math.min((dy - dx) / div, 0)
      return (1 + aboveness)
    end
  end

  local function faceMiddle(x, y)
    local mx, my = (xs[1] + xs[2] + xs[14]) / 3, (ys[1] + ys[2] + ys[14]) / 3
    local md = ( distance(math.abs(xs[1]  - mx), math.abs(ys[1]  - my))
             +   distance(math.abs(xs[2]  - mx), math.abs(ys[2]  - my))
             +   distance(math.abs(xs[14] - mx), math.abs(ys[14] - my))) / 3
    return (2 * md - distance(math.abs(x - mx), math.abs(y - my))) / md
  end

  local rest =
    { {  1, between(xs[3] - cropw, ys[3] - croph, xs[4] - cropw, ys[4] - croph) }
    , {  2, between(xs[5] - cropw, ys[5] - croph, xs[6] - cropw, ys[6] - croph) }
    , {  7, above(xs[3] - cropw, ys[3] - croph) }
    , {  8, above(xs[4] - cropw, ys[4] - croph) }
    , {  9, above(xs[5] - cropw, ys[5] - croph) }
    , { 10, above(xs[6] - cropw, ys[6] - croph) }
    , { 14, between(xs[12] - cropw, ys[12] - croph, xs[13] - cropw, ys[13] - croph) }
    , { 15, between(xs[12] - cropw, ys[12] - croph, xs[13] - cropw, ys[13] - croph) }
    , { 11, faceMiddle }
    }

  for _, set in pairs(rest) do
    local j, f = set[1], set[2]
    mapToPix(map[j], f)
    gnuplot.figure()
    gnuplot.imagesc(map[j])
    local locx, locy, max = getMaxima(map[j])
    temppic[locy][locx] = 10
    xs[j] = locx
    ys[j] = locy
  end

  gnuplot.figure()
  gnuplot.imagesc(temppic)

--else
--  gnuplot.imagesc(0.2126 * pic[1] + 0.7152 * pic[2] + 0.0722 * pic[3])
--end

--local x, y = set[2][i][1], set[2][i][2]
--  gnuplot.figure()
--  gnuplot.plot('', torch.DoubleTensor({set[2][i][1]}), torch.DoubleTensor({set[2][i][2]}))
--gnuplot.raw('set arrow from ' .. x .. ',' .. y-5 .. ' to ' .. x .. ',' .. y+5 .. ' nohead front ls 1')

  if net and false then
    net:evaluate()
    net:forward(pic:reshape(1, pic:size(1), pic:size(2), pic:size(3)):repeatTensor(2,1,1,1))
    net:applyToModules(outputImage)
    forAllModules(net, function(module, level) if module.weight then print(string.rep(' ', level) .. (module.weight and torch.sum(torch.le(module.weight, torch.zeros(module.weight:size()))) or '')); --[[mat2img(module.weight)]] end end)
  end
end
