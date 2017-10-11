local posix = require 'posix'
local bit32 = require 'bit32'

local utils = {}

function utils.hook(table, funcName, newFunc)
  local oldFunc = table[funcName]
  table[funcName] = function(...)
      return newFunc(oldFunc, ...)
    end
end

function utils.preHook(table, funcName, newFunc)
  local oldFunc = table[funcName]
  table[funcName] = function(...)
      newFunc(...)
      return oldFunc(...)
    end
end

function utils.postHook(table, funcName, newFunc)
  local oldFunc = table[funcName]
  table[funcName] = function(...)
      local result = {oldFunc(...)}
      newFunc(...)
      return table.unpack(result)
    end
end

local count = 0
function utils.print10(msg)
  if count < 10 then
    print(msg)
  else
    error()
  end
  count = count + 1
end

function utils.shallowCopyTable(table)
  local copy = {}

  for k, v in pairs(table) do
    copy[k] = v
  end

  return copy
end

function utils.deepCopyTable(table)
  local copy = {}

  for k, v in pairs(table) do
    if type(v) == 'table' then
      local meta = getmetatable(v)
      copy[k] = utils.deepCopyTable(v)
      if meta then setmetatable(copy[k], meta) end
    else
      copy[k] = v
    end
  end

  return copy
end

function utils.tableConcat(table1, table2)
  local result = {}

  for _, v in pairs(table1) do
    result[#result + 1] = v
  end
  for _, v in pairs(table2) do
    result[#result + 1] = v
  end

  return result
end

function utils.tableMerge(table1, table2)
  local result = {}

  for k, v in pairs(table1) do
    result[k] = v
  end
  for k, v in pairs(table2) do
    result[k] = v
  end

  return result
end

-- Create a table with numbers from first to last
function utils.range(first, last)
  local result = {}
  for i = first, last do
    result[#result + 1] = i
  end
  return result
end

function utils.map(f, table)
  for k, v in pairs(table) do
    table[k] = f(table[k])
  end

  return table
end

function utils.iterMap(f, iter, state, var)
  return function()
    local args = table.pack(iter(state, var))
    if args[1] ~= nil then
      var = args[1]
      return f(unpack(args))
    end
  end
end

function utils.repTable(table, n)
  local result = {}
  for i = 1, n do
    result[i] = table
  end
  return result
end

function utils.countEntries(table)
  local count = 0

  for k, v in pairs(table) do
    count = count + 1
  end

  return count
end

-- Get a global variable using a string of its name
-- Example: getByName('os.time') gets the function os.time
function utils.getByName(name)
  local v = _G    -- start with the table of globals
  for w in string.gfind(name, "[%w_]+") do
    v = v[w]
  end
  return v
end

-- Find maximally rectangular width and height such that width*height = len
function utils.findDims(len)
  local w = math.floor(math.sqrt(len))
  local h = w

  while w*h ~= len do
    if w*h < len then
      h = h + 1
    else
      w = w - 1
    end
  end

  return w, h
end

-- Each entry in the input table must be an array-like table
-- (only consecutive numeric indices starting from 1).
-- Picks one value at a time for each entry in input table
-- generating all possible combinations of entries.
-- The entry can also be a function, in which case it must accept one parameter
-- such that when given a nil, it returns the maximum index n,
-- and when given an index (integer from 1..n) it gives a value for the entry.
-- If the 'makeIterator' parameter is true, returns an iterator, otherwise a table.
-- Keys in the iterator/table are the numeric indices of the current variation.
-- Essentially implements the list monad.
function utils.nonDeterministic(table, makeIterator)
  makeIterator = makeIterator == nil and true or makeIterator
  local state = {}
  local lengths = {}
  local variant = nil

  for k, v in pairs(table) do
    state[k] = 1
    lengths[k] = type(v) == 'table' and #v or v(nil)
  end

  local function nextVariant(index)
    if not index then
      return
    end

    if state[index] < lengths[index] then
      state[index] = state[index] + 1
      return true
    end

    state[index] = 1
    index = next(table, index)
    return nextVariant(index)
  end

  local function iterator()
    if variant == nil then
      variant = 1
    else
      variant = variant + 1
      if not nextVariant(next(table, nil)) then
        return
      end
    end

    local current = {}

    for k, v in pairs(table) do
      if type(v) == 'table' then
        current[k] = v[state[k]]
      else
        current[k] = v(state[k])
      end
    end

    return variant, current
  end

  if makeIterator then
    return iterator
  end

  local result = {}
  for k, v in iterator do
    result[k] = v
  end
  return result
end

-- Iterator a -> (a -> Iterator b) -> Iterator b
function utils.bindIter(func, iterator, state, var)
  assert(type(func) == 'function', 'utils.bindIter need a function and an iterator.')
  local subIterator, subState
  local args, subArgs = {}, {}
  args = table.pack(iterator(state, var))
  if args[1] == nil then
    return function() end
  end

  subIterator, subState, subArgs[1] = func(unpack(args))

  local function iterate()
--print(args)
--print(subState)
--print(subArgs)
    subArgs = table.pack(subIterator(subState, subArgs[1]))

    if subArgs[1] == nil then
      args = table.pack(iterator(state, args[1]))
      if args[1] == nil then return end
      subIterator, subState, subArgs[1] = func(unpack(args))
      return iterate()
    end

    return unpack(subArgs)
  end

  return iterate
end

-- Aux
function utils.bindNonDet(ndTable, func)
  local count = 0

  local iterator = utils.iterMap(function(k,v)
      count = count + 1
      return count, v
    end, utils.nonDeterministic(ndTable))

  return utils.bindIter(func, iterator)
end

--[[
function utils.bindNonDet(ndtable, func, makeIterator)
  makeIterator = makeIterator == nil and true or makeIterator
  if not makeIterator then
    local choices = utils.nonDeterministic(ndtable, false)
    local final = {}
    for _, v1 in pairs(choices) do
--print(func(v1))
      final = utils.tableConcat(final, func(v1))
    end
    return final
  end

  local ndIter = utils.iterMap(func, utils.nonDeterministic(ndtable))
  local iterf, state, item = ndIter()
  local count = 0

  return function()
    repeat
      item = iter(state, item)
      if item == nil then
        _, iter = ndIter(iter)
        if not iter then return nil end
      end
    until item ~= nil
    count = count + 1
    return count, item
  end

--  local ndIter = utils.nonDeterministic(ndtable)

--  return utils.iterMap(func, ndIter)
--[[
  local curTopIter, curSubIter, curItem
  local count = 0

  return function()
    repeat
      if not curSubIter then
print('new top iter')
print(curTopIter)
        _, curTopIter = ndIter(curTopIter)
print(curTopIter)
        if not curTopIter then
          return nil
        end
        curSubIter = func(curTopIter)
        curItem = nil
      end
      if type(curSubIter) == 'function' then
print('new sub iter')
print(curSubIter)
        _, curItem = curSubIter(curItem)
print(curSubIter)
      else
        curItem = curSubIter
        curSubIter = nil
      end
    until curItem ~= nil

    count = count + 1
    return count, curItem
  end
end]]

-- Most basic and common validator
function utils.nClassValidator(network, dataset)
  network:forward(dataset[1])

  local _, predictions = torch.max(network.output, 2)
  local correct = torch.sum(torch.eq(predictions:squeeze(), dataset[2]:long()))

  network:clearState()

  return correct / dataset[1]:size(1), predictions
end

-- Validator for hinge loss
-- Considers 0 to be class -1
function utils.hingeValidator(network, dataset)
  network:forward(dataset[1])

  local predictions = torch.gt(network.output, torch.zeros(network.output:size())):long() * 2 - 1
  local correct = torch.sum(torch.eq(predictions:squeeze(), dataset[2]:long()))

  network:clearState()

  return correct / dataset[1]:size(1), predictions
end

-- Display a measure in human readable format
local largePrefixes = {'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'}
local smallPrefixes = {'m', 'u', 'n', 'p', 'f', 'a', 'z', 'y'}

function utils.humanReadable(number, unit, base)
  unit = unit or 'b'
  base = base or 1024
  local scale = 0
  local prefix = ''

  if number < 1 then
    while number < 1 do
      scale = scale + 1
      number = number * base
    end
    prefix = smallPrefixes[scale]
  elseif number > base then
    while number > base do
      scale = scale + 1
      number = number / base
    end
    prefix = largePrefixes[scale]
  end

  return number .. ' ' .. prefix .. unit
end

-- Copied from a stackexchange answer by Arrowmaster
-- Changed \\ to / to make it work on linux
function utils.parsePath(path)
  return string.match(path, "(.-)([^/]-([^%.]+))$")
end

assert( posix.isatty( posix.STDIN_FILENO ), "stdin not a terminal" )
local saved_tcattr = assert( posix.tcgetattr( posix.STDIN_FILENO ) )
local raw_tcattr = utils.deepCopyTable( saved_tcattr )

raw_tcattr.lflag = bit32.band( raw_tcattr.lflag, bit32.bnot( posix.ICANON ) )

local guard = setmetatable( {}, { __gc = function()
  posix.tcsetattr( posix.STDIN_FILENO, posix.TCSANOW, saved_tcattr )
end } )


function utils.kbhit()
  assert( posix.tcsetattr( posix.STDIN_FILENO, posix.TCSANOW, raw_tcattr ) )
  local r = assert( posix.rpoll( posix.STDIN_FILENO, 0 ) )
  assert( posix.tcsetattr( posix.STDIN_FILENO, posix.TCSANOW, saved_tcattr ) )
  return r > 0
end

return utils
