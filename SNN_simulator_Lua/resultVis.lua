require 'nn'
require 'optim'
require 'LargeMarginCriterion'
require 'gnuplot'

local lfs = require 'lfs'

local resultDir = '/worktmp/newest results' --'./results'

for file in lfs.dir(resultDir) do
  if file ~= '.' and file ~= '..' then
    local path = resultDir .. '/' .. file
    local content = torch.load(path, 'ascii')

    if --[[content.config.config.desc.lrateDecay
    and not content.config.config.desc.residual
    and not content.config.config.desc.dropout
    and content.config.config.desc.init == 'kaiming'  then --]] --content.config.accuracy >= 0.98 then
      --content.config.config.desc.residual and math.random(16) == 1 then
      file == 'config55.cfg' then
      print(file)
      gnuplot.figure()
      gnuplot.plot(file, torch.range(1, 300), content.valAcc, '-')
      print(content.config.config.desc)
      print(content.config.accuracy)
    end
  end
end
