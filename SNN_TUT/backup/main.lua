--[[
Open rasterplot and print visually the content
]]--

local f = io.open("../exc.1.ras", "rb")
local tab = {}
local row = {}
local draw_exc = {}

local time = 0
local timestep = 0.0167 * 20

local time_val = 0
local id_val = 0
local size = 3

local s_w = 0

local acquisition = true
local index = 1

local max_val = 0

local exist = false

function love.load()

    s_w = love.graphics.getWidth()

    while acquisition do
        time_val = f:read "*n"
        id_val = f:read "*n"

        if time_val == nil then
            acquisition = false
        else
            if time_val < time + timestep then
                table.insert(row, id_val)
            else
                time = time + timestep
                table.insert(tab, row)
                row = {}
                table.insert(row, id_val)
            end
        end
    end
    --[[ at this point there is a multi array "tab"
    storing every spking neurons for each 0.1s of simulation]]--
end

function love.update()
end

function love.draw()
    love.graphics.clear()
    -- spikes
    love.graphics.setColor(255, 255, 150)
    love.graphics.print(index * timestep)
    for k, v in ipairs(tab[index]) do
        local x = math.floor(v/128) * size * 4 + s_w / 2 - size * 128
        local y = (v % 128) * size + 200
        print(v)
        love.graphics.rectangle("fill",x, y, size*2, size*2)
    end
    index = index + 1
end
