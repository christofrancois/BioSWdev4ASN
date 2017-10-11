--[[
Open rasterplot and print visually the content
Param defines the intensity of the averaging

require love2d installed
run :
  > love ./
]]--

--local f = io.open("../../results/55->0.e.ras", "rb")
local f = io.open("./55->0.e.ras", "rb")
local size = 4096*2
local param = 3

local tab = {}
local mean = {}
local row = {}
local draw_exc = {}

local speed = 1
local time = 0
local timestep = 0.0167 * 1.5
local time_val = 0
local id_val = 0
local index = 1

local s_w = 0
local draw_size = 3

local run = false

function love.load()
    s_w = love.graphics.getWidth()

    acquisition()
end

function love.draw()
  if run then
    love.graphics.clear()
    love.graphics.setColor(255, 255, 150)
    love.graphics.print("Time : "..index * timestep,10,10,0,2,2)
    print(index)
    for i = 1, size do
      --print(index.." "..i)
      if tab[index] == nil then
        run = false
        return
      end
      local a = tab[index][i] * 255
      local x = math.floor((i-1)/128) * draw_size * 4 + s_w / 2 - draw_size * 128
      local y = ((i-1) % 128) * draw_size*1 + 150
      love.graphics.setColor(255, 255, 150, a)
      love.graphics.rectangle("fill",x, y, draw_size*3, draw_size*3)
    end
    index = index + 1
  end
end

function freeTable(n)
    t = {}
    for i=1,n do
        t[i] = 0
    end
    return t
end

function acquisition()
    local i = 1
    for x = -param, param do
      tab[i + x] = freeTable(size)
    end
    while true do
        time_val = f:read "*n"
        if time_val == nil then return end
        id_val = (f:read "*n") + 1

        if time_val < time + timestep then
            for x = -param, param do
              tab[i + x][id_val] = tab[i + x][id_val] + 1 / (2* param + 1)
            end
        else
            i = i + 1
            tab[i + param] = freeTable(size)
            time = time + timestep
        end
    end
end

function love.keypressed(key)
  run = true;
end
