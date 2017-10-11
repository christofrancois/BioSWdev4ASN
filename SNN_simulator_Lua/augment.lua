-- Data augmentations
-- For now only for images

local augment = {}

local make3d = function(mat)
  if mat:dim() == 2 then
    mat = mat:reshape(1, mat:size(1), mat:size(2))
  elseif mat:dim() ~= 3 then
    error( 'Cannot augment dataset of dimensionality ' .. mat:dim() .. '.\n'
        .. 'Please provide a single 2D sample, or a 3D set of samples.')
  end
end

-- Horizontal flip
function augment.hflip(dataset)
  local mat = dataset[1]
  make3d(mat)

  local unflipped = mat:clone()
  local temp

  for i = 1, mat:size(1) do
    for j = 1, math.floor(mat:size(2) / 2) do
      temp = mat[i][{{}, j}]
      mat[i][{{}, j}] = mat[i][{{}, mat:size(2) - j + 1}]
      mat[i][{{}, mat:size(2) - j + 1}] = temp
    end
  end

  local trg = dataset[2]

  return { torch.cat(unflipped, mat, 1), torch.cat(trg, trg, 1) }
end

-- Crop 5 images from mat each half the original in all dimensions
-- One image is from the center, others are from the corners
function augment.crop5(dataset, w, h)
  local mat = dataset[1]

  make3d(mat)

  w = w or math.ceil(mat:size(3) / 2)
  h = h or math.ceil(mat:size(2) / 2)

  local pic1 = mat:narrow(2, 1, h)
  local pic2 = pic1:narrow(3, mat:size(3) - w + 1, w)
  pic1 = pic1:narrow(3, 1, w)

  local pic3 = mat:narrow(2, math.floor(mat:size(2) / 2 - h / 2), h)
  pic3 = pic3:narrow(3, math.floor(mat:size(3) / 2 - w / 2), w)

  local pic4 = mat:narrow(2, mat:size(2) - h + 1, h)
  local pic5 = pic4:narrow(3, mat:size(3) - w + 1, w)
  pic4 = pic4:narrow(3, 1, w)

  local trg = dataset[2]

  return { torch.cat({ pic1, pic2, pic3, pic4, pic5 }, 1)
         , torch.cat({ trg, trg, trg, trg, trg }, 1)
         }
end

function augment.zca(dataset)
  require 'unsup'

  local oldsize = dataset[1]:size()

  return { torch.reshape(unsup.zca_whiten(dataset[1]:clone()), oldsize), dataset[2] }
end

-- Creates jittered images from an image
-- jitter amounts are from the range [-grade, -grade + 1, .., grade - 1, grade] * stride
-- each jitter amount is taken as an extra datapoint with the given probability
function augment.jitter(dataset, grades, stride, probability)
  local originalSize = dataset[1]:size(1)
  local w = dataset[1]:size(3)
  local h = dataset[1]:size(4)
  for i = -grades, grades do
    for j = -grades, grades do
      if i~= 0 or j ~= 0 then
        local left = math.max(i * stride + 1, 1)
        local right = math.min(i * stride + w, w)
        local top = math.max(j * stride + 1, 1)
        local bot = math.min(j * stride + h, h)
        for n = 1, originalSize do
          if torch.bernoulli(probability) == 1 then
            local newpic = dataset[1][n]:clone():reshape(1, dataset[1]:size(2), w, h)
            newpic[1][{{}, {w - right + 1, w - left + 1}, {h - bot + 1, h - top + 1}}] =
              dataset[1][n][{{}, {left, right}, {top, bot}}]
            dataset = { torch.cat(dataset[1], newpic, 1), torch.cat(dataset[2], torch.zeros(1):fill(dataset[2][n]), 1) }
          end
        end
      end
    end
  end

  return dataset
end

-- Gaussian noise
-- mean and std are currenlty ignored
function augment.gnoise(dataset, number, retain, multiplier, mean, std)
  mean = mean or 0
  std = std or 1
  number = number or 1
  multiplier = multiplier or 1
  if retain == nil then retain = true end

  local newdata = retain and dataset[1]:clone() or dataset[1]
  newdata = torch.repeatTensor(newdata, number, 1, 1, 1)
  newdata:add(torch.randn(newdata:size()) * multiplier)

  if retain then
    dataset[1] = torch.cat(dataset[1], newdata, 1)
    dataset[2] = torch.repeatTensor(dataset[2], number + 1)
  else
    dataset[1] = newdata
    dataset[2] = torch.repeatTensor(dataset[2], number)
  end

  return dataset
end

-- Salt and pepper noise
function augment.saltAndPepper(dataset, probability, number, retain)
  probability = probability or 0.25
  number = number or 1
  if retain == nil then retain = true end

  local newdata = retain and dataset[1]:clone() or dataset[1]
  newdata = torch.repeatTensor(newdata, number, 1, 1, 1)
  local mask = torch.le(torch.rand(newdata:size()), probability):double()
  local noise = torch.le(torch.rand(newdata:size()), 0.5):double()
  newdata = torch.cmul(1 - mask, newdata) + torch.cmul(mask, noise)

  if retain then
    dataset[1] = torch.cat(dataset[1], newdata, 1)
    dataset[2] = torch.repeatTensor(dataset[2], number + 1)
  else
    dataset[1] = newdata
    dataset[2] = torch.repeatTensor(dataset[2], number)
  end

  return dataset
end

function augment.zoom(dataset, magnitude, number, retain)
  magnitude = magnitude or 0.15
  number = number or 1
  if retain == nil then retain = true end

  local newdata = retain and dataset[1]:clone() or dataset[1]
  newdata = torch.repeatTensor(newdata, number, 1, 1, 1)

  require 'image'

  for i = 1, newdata:size(1) do
    local mag = torch.uniform(1 - magnitude, 1 + magnitude)
    local w, h = math.floor(mag * newdata:size(3)), math.floor(mag * newdata:size(4))
    if mag < 1 then
      local x, y = math.ceil((newdata:size(3) - w) / 2), math.ceil((newdata:size(4) - h) / 2)
--print(x,y, w, h)
      newdata[i][{{}, {x, x + w - 1}, {y, y + h - 1}}] = image.scale(newdata[i], w, h)
    else
      newdata[i] = image.crop(image.scale(newdata[i], w, h), 'c', newdata:size(3), newdata:size(4))
    end
  end

  if retain then
    dataset[1] = torch.cat(dataset[1], newdata, 1)
    dataset[2] = torch.repeatTensor(dataset[2], number + 1)
  else
    dataset[1] = newdata
    dataset[2] = torch.repeatTensor(dataset[2], number)
  end

  return dataset
end

-- RGB to grayscale
function augment.grayscale(dataset)
  local oldsize = dataset[1]:size()
  dataset[1] = 0.2126 * dataset[1][{{}, 1}]
             + 0.7152 * dataset[1][{{}, 2}]
             + 0.0722 * dataset[1][{{}, 3}]

  oldsize[2] = 1
  dataset[1] = dataset[1]:reshape(oldsize)

  return dataset
end

function augment.normalize(dataset, wholeset)
  if wholeset == nil then wholeset = true end

  if wholeset then
    local mean, std = torch.mean(dataset[1]), torch.std(dataset[1])

    return { (dataset[1] - mean) / std, dataset[2] }, mean, std
  end

  for x = 1, dataset[1]:size(4) do
    for y = 1, dataset[1]:size(3) do
      local pixelset = dataset[1][{{}, {}, y, x}]
      local mean, std = torch.mean(pixelset), torch.std(pixelset)
      dataset[1][{{}, {}, y, x}] = (dataset[1][{{}, {}, y, x}] - mean) / std
    end
  end

  return dataset 
end

-- Some auxiliary function to modify dataset targets for certain tasks

-- Set targets to 0 or 1 for <=0, >0 respectively
function augment.binaryTargets(dataset)
  local targets = torch.gt(dataset[2]:double(), torch.zeros(dataset[2]:size())):double()

  return {dataset[1], targets}
end

-- Set targets to -1 or 1 for <=0, >0 respectively
function augment.hingeTargets(dataset)
  dataset = augment.binaryTargets(dataset)
  dataset[2] = dataset[2] * 2 - 1

  return dataset
end

return augment
