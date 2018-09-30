--[[
  this is a logger that combines a number of other loggers

  it implements the LoggerInterface

  attention: when using a composit logger,
  one of the loggers can change its call arguments in-place
  and the other loggers will no longer get the original data
  this happens in particular if "transformOutputAndTarget" is used
  in LoggerClassification
--]]

local LoggerComposit= 
  torch.class('NetworkTraining.LoggerComposit')

function LoggerComposit:__init(loggers)
--[[
  loggers is a table of objects implementing the LoggerInterface
--]]
  self.loggers=loggers
  for k,v in pairs(self.loggers) do
    NetworkTraining.check_object_interface_compatibility
      (NetworkTraining.LoggerInterface,v)
  end
end

function LoggerComposit:addToLog(output,target,input,f,dfdx)
  for k,v in pairs(self.loggers) do
    v:addToLog(output,target,input,f,dfdx)
  end
end

function LoggerComposit:logEpochEnd(net)
  for k,v in pairs(self.loggers) do
    v:logEpochEnd(net)
  end
end

function LoggerComposit:logUpdate(net,optimState,params,gradParams)
  for k,v in pairs(self.loggers) do
    v:logUpdate(net,optimState,params,gradParams)
  end
end

