------------------------------------------------------------------------
--[[ ImageFeedback ]]--
-- Feedback
-- Displays the output tensor as an image
-- 
------------------------------------------------------------------------
local ImageFeedback, parent = torch.class("dp.ImageFeedback", "dp.Feedback")
ImageFeedback.isImageFeedback = true

function ImageFeedback:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], 
      "Constructor requires key-value arguments")
   local args, width, channels, name, freq = xlua.unpack(
      {config},
      'imageFeedback', 
      'Displays the output tensor as an image',
      {arg='width', type='number',
       help='Width of the image'},
      {arg='channels', type='number', default=1,
       help='number of color channels'},
      {arg='name', type='string', default='imagefeedback',
       help='name identifying Feedback in reports'},
      {arg='frequency', type='number', default=5,
       help='Display images every this many batces'}
   )
   config.name = name
   self._width = width
   self._channels = channels
   self._imagedata = {}
   self._frequency = freq
end

function ImageFeedback:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

function ImageFeedback:_add(batch, output, report)
  
  self._imagedata = output
end

function ImageFeedback:display()
  require 'image'
  local size = self._imagedata:size(1)
  for i = 1,size,1 do
    local imgt = image.toDisplayTensor(
        torch.reshape( self._imagedata[i]
                     , self._channels
                     , self._width
                     , self._imagedata[i]:nElement()/self._width
                     ))
    image.display(imgt)
  end
end

function ImageFeedback:doneEpoch(report)
  --[[require 'gm'
  local image = gm.Image(
      torch.reshape( self._imagedata
                   , self._channels
                   , self._width
                   , self._imagedata:nElement()/self._width
                   )
    , 'I'
    , 'DWH')
  image:save('res.png')]]
  if report.epoch % self._frequency == 0 then
    self:display()
  end
end

function ImageFeedback:_reset()

end

function ImageFeedback:report()
   return {}
end
