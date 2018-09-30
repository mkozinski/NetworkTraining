
--[[ 
  implements the LoggerInterface
  this logger is to be used when ground truth has the form of class indexes
  the target (ground truth) tensor can have various shapes
  for example for image classification and image segmentation 

  it creates a log file
  ...Epoch.log   updated every time a training epoch is finished
                   general accuracy, average per-class recall,
                   average intersection over union,
--]]

require "optim"
local LoggerClassification= 
  torch.class('NetworkTraining.LoggerClassification')

function LoggerClassification:__init(loggerConf)
  --[[
  loggerConf can have the following members
  logdir
  prefix
  nClasses   the number of classes in the classification problem
  transformOutputAndTarget
             (optional) a function for remapping the input and target
             can be useful e.g. for removing the ignored value
             or to remap if the network and targets are defined in terms of 
             different labels
  saveBest   if set, best networks are saved
  --]]
  self.logdir=loggerConf.logdir
  self.confusion=optim.ConfusionMatrix(loggerConf.nClasses)

  -- the epochLogger is used for writing information that is collected
  -- for each forward-backward and written once per epoch
  self.prefix=loggerConf.prefix or ""
  self.epochName=self.prefix.."Epoch.log" -- name of the per-epoch logger
  self.epochLogger=optim.Logger(paths.concat(self.logdir,self.epochName))
  self.epochLogger:setNames{'acc', 'avg per-class recall', 'avg iou'} 
  self.epochLogger.showPlot = false

  self.bestAcc=0
  self.bestAvgPCR=0 --best average per class recall
  self.bestIoU=0
  
  self.transformOT=
    loggerConf.transformOutputAndTarget or function (o,t) return o,t end
  self.saveBest=loggerConf.saveBest
end

function LoggerClassification:addToLog(output,target,input,f,dfdx)
  -- the input is ignored
  if output and target then 
    local o,t=self.transformOT(output,target)
    if t:gt(0):sum()>0 then self.confusion:batchAdd(o,t) end
  end
end

function LoggerClassification:logEpochEnd(net)
  self.confusion:updateValids()
  local acc=self.confusion.totalValid*100
  local acr=self.confusion.valids:mean()*100
  local aiu=self.confusion.averageUnionValid*100
  self.epochLogger:add{acc,acr,aiu}

  if net and self.saveBest then 
    net:clearState()
    if acc>self.bestAcc then
      self.bestAcc=acc
      torch.save(paths.concat(self.logdir,self.prefix..'_net_bestAcc.t7'),net)
      torch.save(paths.concat(self.logdir,self.prefix.."Confusion_bestAcc.t7"),
             self.confusion)
    end
    if acr>self.bestAvgPCR then
      self.bestAvgPCR=acr
      torch.save(paths.concat(self.logdir,self.prefix..'_net_bestAPCR.t7'),net)
      torch.save(paths.concat(self.logdir,self.prefix.."Confusion_bestAPCR.t7"),
             self.confusion)
    end
    if aiu>self.bestIoU then
      self.bestIoU=aiu
      torch.save(paths.concat(self.logdir,self.prefix..'_net_bestIoU.t7'),net)
      torch.save(paths.concat(self.logdir,self.prefix.."Confusion_bestIoU.t7"),
             self.confusion)
    end
  end
  torch.save(paths.concat(self.logdir,self.prefix.."Confusion.t7"),
             self.confusion)
  self.confusion:zero()
end

function LoggerClassification:logUpdate()
end
