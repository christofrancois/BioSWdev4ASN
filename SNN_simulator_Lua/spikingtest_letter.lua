package.path = package.path .. ';?.lua'

require 'spikingreservoir_rk'
require 'gnuplot'
local models = require 'models'
local numint = require 'numint'
local ProFi = require 'ProFi'

local profile = false

if profile then
  ProFi:start()
end

local file = assert(io.open('data.dat', 'w'))

local interp = {}
local textInput
local nChars = 1
do
  local inputFile = assert(io.open('msnd.txt', 'r'))

  textInput = inputFile:read("*all")
  local set = {}

  textInput:gsub(".", function(c)
      set[c] = true
    end)

  nChars = 1
  for k, _ in pairs(set) do
    interp[k] = nChars
    interp[nChars] = k
    nChars = nChars + 1
  end
end

local spikeN = 0
local timeScale = 1
local timeStep = 1e-3*timeScale -- s / step
local testLength = 10 -- s
local exciteSteps = testLength / timeStep--math.floor(testLength / 2 + 0.5)
local restSteps = 0--math.floor(testLength / 6 + 0.5)
local relearnSteps = 0-- math.floor(testLength / 6 * 2 + 0.5)
local totalSteps = exciteSteps + restSteps + relearnSteps
local window = 200
local networkScale = 3
local inputs = nChars
local outputs = nChars
local inspike = 1--/timeStep
local infreq = 0.2/timeScale*10
local printEvery = 1
local inputLag = 1

local sr = nn.SpikingReservoir
  { neurons = {models.regularSpiking(700*networkScale,{
      C = models.constant(100)
    , k = models.constant(0.7)--3)--0.7)
    , v_r = models.constant(-60)
    , v_t = models.constant(-40)
    , v_peak = models.constant(35)
    , a = models.constant(0.13)--0.03)--0.13)
    , b = models.constant(-2)
    , c = models.constant(-50)
    , d = models.constant(400)
    , tau_x = models.constant(100)
    , p = models.constant(0.65)
    , g_AMPA = models.constant(1)
    , g_NMDA = models.constant(1)
    , g_GABAA = models.constant(0)
    , g_GABAB = models.constant(0)
    }), models.fastSpiking(280*networkScale), models.latentSpiking(20*networkScale)}
  , connectivity = 0.15
  , inputs = torch.range(1, inputs + outputs):long()
  , outputs = torch.range(1 + inputs, inputs + outputs):long()
  , spikeCallback = function(x, time) file:write(time .. ' ' .. x .. '\n'); spikeN = spikeN + 1 end
  , timeStep = timeStep
  , subSteps = 5/timeScale
  , n = 3
  , model = numint.euler
  , params = {pre_tau = 15, post_tau = 15, a_minus = 0.1, a_plus = 0.1}
  , inputType = 'spikes'
  }


--sr:cuda()

--[[remove latent-latent connections
local latentStart = 950*networkScale
local latentEnd = 1000*networkScale
sr._connections:sub(latentStart, latentEnd, latentStart, latentEnd):fill(0)
sr._weights:sub(latentStart, latentEnd, latentStart, latentEnd):fill(0)]]

local potentials = torch.zeros(totalSteps, sr._nTotal)
local spikes1 = torch.zeros(window, outputs)
local spikes2 = torch.zeros(window, outputs)
local spikes3 = torch.zeros(window, outputs)
local monitoreds = {inputs, inputs + 1}
local monitorVars = {'_potential', '_gAMPA', '_gNMDA', '_gGABAA', '_gGABAB', '_recovery', '_preTrace'}
local monitors = {}

for _, m in pairs(monitoreds) do
  monitors[m] = {}
  for _, v in pairs(monitorVars) do
    monitors[m][v] = torch.zeros(totalSteps)
  end
end

local function updMonitors(i)
  for _, m in pairs(monitoreds) do
    for _, v in pairs(monitorVars) do
      monitors[m][v][i] = sr[v][m]
    end
  end
end

--gnuplot.imagesc(sr._weights ,'color')

print('running simulation, nChars: ' .. nChars)
local time1 = os.time()

local function encode(char, freq)
  local z = torch.zeros(nChars)
  --print(interp[char])
  z[interp[char]] = freq
  return z
end

for i = 1, exciteSteps, 1 do
  local textPos = i % textInput:len()
  local textPredPos = (i - inputLag + 1) % textInput:len()
  sr:forward(torch.cat(encode(textInput:sub(textPos, textPos), infreq), encode(textInput:sub(textPredPos, textPredPos), infreq)) * inspike)
  --potentials[i] = sr._potential

  updMonitors(i)
  if i % printEvery == 0 then print(i .. '/' .. totalSteps .. ', ' .. spikeN .. ' spikes, ' .. torch.sum(spikes1) / (timeStep * window * outputs) .. ' average output spiking frequency (Hz)') end
end

if profile then
  ProFi:stop()
  ProFi:writeReport('spikeProf.txt')
end

print('simulation ended')
print('ran ' .. (totalSteps * sr._timeStep) / 1000 .. ' seconds of simulation in ' .. os.time() - time1 .. ' seconds')

file:flush()
file:close()

sr._spikeCallback = function()end

--torch.save('spikenet.net', sr, 'ascii')

gnuplot.figure()
gnuplot.raw("plot 'data.dat' with dots")

print('Interpretations')
for k, v in pairs(interp) do
  print(k, v)
end

local outByte = nil
print('Trying to write stuff...')

for i = 1, exciteSteps, 1 do
  local textPos = i % textInput:len()
--  local textPredPos = (i - inputLag + 1) % textInput:len()
  sr:forward(torch.cat(encode(outByte or " ", infreq), torch.zeros(outputs)))
  local y, j = torch.max(sr.output, 1)
  outByte = interp[j[1]]
  if outByte then
    io.stdout:write(outByte)
    io:flush()
  end
end
print(' ')
print('total ' .. spikeN .. ' spikes')

--[[
local spikerates = torch.sum(spikes1, 1) / (window * sr._timeStep)
gnuplot.figure()
gnuplot.imagesc(spikerates:reshape(5,5) / 1000)
spikerates = torch.sum(spikes2, 1) / (window * sr._timeStep)
gnuplot.figure()
gnuplot.imagesc(spikerates:reshape(5,5) / 1000)
spikerates = torch.sum(spikes3, 1) / (window * sr._timeStep)
gnuplot.figure()
gnuplot.imagesc(spikerates:reshape(5,5) / 1000)
]]
--[[
for k, m in pairs(monitoreds) do
  for _, v in pairs(monitorVars) do
    gnuplot.figure()
    gnuplot.plot(v .. '_' .. k, torch.range(0, totalSteps - 1) * timeStep, monitors[m][v], '-')
  end
end
]]
--[[
gnuplot.figure()
gnuplot.imagesc(potentials:t() ,'color')
gnuplot.figure()
gnuplot.imagesc(sr._weights ,'color')
gnuplot.figure()
gnuplot.imagesc(sr._x)]]

