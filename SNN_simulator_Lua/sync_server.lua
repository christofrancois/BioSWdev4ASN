-- Receives files from remote location
-- Makes backups of old files and refuses to receive files whose backups have not been removed
-- (i.e. remove backups to commit)
-- Only allows syncing files, e.g. no new files are allowed, only new version of old files
-- No slashes ('/') are allowed in the filepath
-- Essentially I do not want this program to be able to mess things around too much

local socket = require 'socket'
local lfs = require 'lfs'

local utils = require 'utils'
local config = arg[1] or 'sync.cfg'
print(arg)
local syncSettings = dofile(config)

local server = socket.bind(syncSettings.address, syncSettings.port)
server:settimeout(syncSettings.serverTimeout)

local function recvData(socket, chunkSize)
  -- Receive required data
  local bytes, errMsg = socket:receive()

  bytes = tonumber(bytes)

  if bytes then
    print('Receiving ' .. utils.humanReadable(bytes) .. ' of data')
  else
    error('Error receiving data: ' .. errMsg)
  end

  local dataLeft = bytes

  return function()
    if dataLeft <= 0 then
      return nil
    end
--print(math.min(chunkSize, dataLeft))
    local chunk, errMsg = socket:receive(math.min(chunkSize, dataLeft))

    if not chunk then
      error('Error receiving data: ' .. errMsg)
    end

    dataLeft = dataLeft - chunkSize
    return chunk
  end
end

local function saveFile(fname, receiver)
  local slash = fname:find('/')
  if slash then
    print('Refusing to create non-local file')
  end

  local file = io.open(fname, 'r')

  if file then
    local backup = lfs.attributes(fname .. '.old')
    file:close()

    if backup then
      print('Backup has not been deleted')
    else
      --local oldContent = file:read('*all')
      --backup = io.open(fname .. '.old', 'w')
print('mv ' .. fname .. ' ' .. fname .. '.old')
      os.execute('mv ' .. fname .. ' ' .. fname .. '.old')
      --if backup then
        --backup:write(oldContent)
      --  backup:close()
        file = io.open(fname, 'w')
        if file then
          for chunk in receiver do
            file:write(chunk)
          end
          file:close()
        else
          print('Could not open target file for writing')
        end
      --else
      --  print('Could not open backup file for writing')
      --end
    end
  else
    print('Refusing to create new file')
  end
end

while true do
  local client = server:accept()

  if client then
    client:settimeout(syncSettings.clientTimeout)

    -- Receive number of files
    local nFiles, errMsg = client:receive()

    if nFiles then
      print('Receiving ' .. nFiles .. ' files')
      local flen, fname, file
      for i = 1, nFiles do
        -- Receive file name
        fname, errMsg = client:receive()
        if fname then
          print('Receiving file ' .. fname)
          local receiver = recvData(client, 1024*1024)
          -- Receive file length
          --[[flen, errMsg = client:receive()
          if flen then
            print('Receiving ' .. flen .. ' bytes')
            file, errMsg = client:receive(flen)]]
            --if file then
            saveFile(fname, receiver)
            --else
            --  print('Error receiving file ' .. fname .. ': ' .. errMsg)
            --end
          --else
          --  print('Error receiving file ' .. fname .. ': ' .. errMsg)
          --end
        else
          print('Error receiving a file: ' .. errMsg)
        end
      end
    end

    if errMsg then
      print('Error while syncing: ' .. errMsg)
    end

  end
end
