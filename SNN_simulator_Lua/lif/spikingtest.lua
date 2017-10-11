require 'spikingreservoir'
require 'gnuplot'

local file = assert(io.open('data.dat', 'w'))
local potentials = torch.zeros(10000,1000)

local i = 0

local sr = nn.SpikingReservoir
  { nExcitatory = 800
  , nInhibitory = 200
  , connectivity = 0.02
  , inputs = torch.LongTensor{1,2,3,4,5,6}
  , outputs = torch.LongTensor{7,8,9}
  , spikeCallback = function(x, time) file:write(time .. ' ' .. x .. '\n'); i = i + 1 end
  , timeStep = 0.001
  , subSteps = 1
  }
--print(sr._connections)
--print(sr._weights)
for i = 1,5000,1 do
  sr:forward(torch.Tensor{5,10,3,2,5,7})
  potentials[i] = sr._potential
end
for i = 5001,10000,1 do
  sr:forward(torch.Tensor{0.1,0.1,0.5,0.1,0.05,0.01})
--  sr:forward(torch.Tensor{0,0,0,0,0,0})
  potentials[i] = sr._potential
end

file:flush()
file:close()

gnuplot.raw("plot 'data.dat' with dots")
gnuplot.figure()
gnuplot.imagesc(potentials:t() ,'color')

print('total ' .. i .. ' spikes')
