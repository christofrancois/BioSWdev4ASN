-----------------------------------------------------------------------
--[[ ConstantPP ]]--
-- A Preprocess that overrides previous data
-- with the given new data
-----------------------------------------------------------------------
local Autoencode = torch.class("dp.Autoencode", "dp.Preprocess")
Autoencode.isAutoencode = true

function Autoencode:__init(new_data)
   assert(new_data.isDataView, "Constructor requires a dataview")
   self._data = new_data
end

function Autoencode:apply(dv)
   assert(dv.isDataView, "Expecting a DataView instance")
   local data = dv:forward('bf')
   dv:replace('bf', self._data)
end
