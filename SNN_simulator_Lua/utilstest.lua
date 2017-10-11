local utils = require 'utils'

local entries =
  { abc = {3, 4, 6, 23}
  , def = {5, 21, 65}
  }

for k, v in utils.nonDeterministic(entries) do
  print(k, v)
end

print('As a table')

print(utils.nonDeterministic(entries, false))

print('Testing iterMap')

for k, v in utils.iterMap(function(k, x) x.jkl = x.abc + x.def return k, x end, utils.nonDeterministic(entries)) do
  print(k, v)
end

print('Testing bind')

local bind = utils.bindIter

for k, v in bind(function(k, v) return utils.iterMap(function(_, v) return k, v end, pairs(v)) end, pairs(entries)) do
  print(k, v)
end

--[[
print(utils.bindNonDet(entries, function(x) return {utils.tableMerge(x, { ghi = x.abc * 2, jkl = x.def + 1 })} end, false))

print(utils.bindNonDet(entries, function(x)
  return utils.bindNonDet({ ghi = { x.abc * 2, x.def + 1 }}, function(y) return {utils.tableMerge(x, y)} end, false) end, false))

print((function(x)
  return utils.bindNonDet({ ghi = { x.abc * 2, x.abc + 1 }}, function(y) return {utils.tableMerge(x, y)} end, false) end)({abc=9}))


print('Testing bind iterator')

for k, v in utils.bindNonDet(entries, function(x) print('outer');
  return utils.bindNonDet({ ghi = { x.abc * 2, x.def + 1 }}, function(y) print('inner'); return pairs({utils.tableMerge(x, y)}) end) end) do
  print(k, v)
end]]
