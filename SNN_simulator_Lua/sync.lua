-- Sends a sync request
local args = arg

local socket = require 'socket'

print(args)
print(#args)

local ip = args[1]
local port = args[2]

local client, errMsg = socket.connect(ip, port)

if client then
  -- Send number of files
  client:send((#args - 2) .. '\n')

  for i = 3, #args do
    local file = io.open(args[i], 'r')
    if file then
      local content = file:read('*all')
      file:close()

      -- Send filename
      client:send(args[i] .. '\n')
      -- Send file length
      client:send(content:len() .. '\n')
      -- Send file
      client:send(content)
    end
  end

  client:close()
else
  error(errMsg)
end
