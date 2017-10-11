require 'nn'

local network = torch.deserialize(torch.load('networks/face_pi.lua.out', 'ascii'), 'ascii')

local trainset = torch.load('larviTrainAug.th7', 'ascii')

network:remove()
network:remove()
network:forward(trainset[1])

torch.save('larviTrainAugFeat.th7', { network.output, trainset[2] }, 'ascii')
