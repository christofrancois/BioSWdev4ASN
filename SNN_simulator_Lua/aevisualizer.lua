require 'gnuplot'

local aeviz = {}

function aeviz.visualize(weights, width, height, margin, normalizeImages, normalizeResult, margincolor)
  if not width and not height then
    error('Visualizer requires at least one dimension (width, height) of input images')
  end

  height = height or math.floor(weights:size(2) / width)
  width = width or math.floor(weights:size(2) / height)
  margin = margin or 5
  margincolor = margincolor or 0.5

  if weights:size(2) ~= height * width then
    error('width * height of images should equal the number of weights per unit')
  end

  local totalImages = weights:size(1)
  local imagesPerRow = math.floor(math.sqrt(totalImages))
  local totalWidth = imagesPerRow * (width + margin) + margin

  local function halves(y, x)
    return torch.zeros(y, x):fill(margincolor)
  end

  local finalImage = halves(margin, totalWidth)
  local imageRow = halves(height, margin)

  local i = 0
  for j = 1, totalImages do
    local image = weights[{j, {}}]

    image:div(image, math.sqrt(torch.sum(torch.pow(image, 2))))
    image = image:reshape(height, width)

    if normalizeImages then
      local min = torch.min(image)
      local max = torch.max(image)
      image = (image - min) / (max - min)
    end

    imageRow = torch.cat(imageRow, image, 2)
    imageRow = torch.cat(imageRow, halves(height, margin), 2)

    i = i + 1
    if i >= imagesPerRow then
      i = 0
      finalImage = torch.cat(finalImage, imageRow, 1)
      finalImage = torch.cat(finalImage, halves(margin, totalWidth), 1)
      imageRow = halves(height, margin)
    end
  end

  if i > 0 then
    imageRow = torch.cat(imageRow, halves(height, (imagesPerRow - i) * (width + margin)), 2)
    finalImage = torch.cat(finalImage, imageRow, 1)
    finalImage = torch.cat(finalImage, halves(margin, totalWidth), 1)
  end

  if normalizeResult then
    local min = torch.min(finalImage)
    local max = torch.max(finalImage)
    finalImage = (finalImage - min) / (max - min)
  end

  gnuplot.imagesc(finalImage)
end

return aeviz
