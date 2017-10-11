-- Quick and dirty network test for trained networks

require 'nn'
require 'sortlayer'
local utils = require 'utils'
local augment = require 'augment'

cmd = torch.CmdLine()
cmd:option('--dataset', '', 'which dataset to use. Must be a torch data file.')
cmd:option('--network', '', 'network to try')
opt = cmd:parse(arg or {})

local dataset = torch.load(opt.dataset, 'ascii')
local network = torch.load(opt.network, 'ascii')

local mean, std = torch.mean(dataset[1]), torch.std(dataset[1])

dataset[1] = (dataset[1]:double() - mean) / std

print(mean, std)

dataset[2] = dataset[2]:double() + 1

local classes = torch.max(dataset[2])
local accuracy, predictions = utils.nClassValidator(network, dataset)

print(accuracy)
print(torch.histc(predictions:double(), classes))
print(torch.histc(dataset[2], classes))
