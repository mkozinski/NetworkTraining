--[[ 
  implements the LoggerInterface
  logs the F1 score, useful for binary segmentation
--]]

require "optim"
local LoggerF1= 
  torch.class('NetworkTraining.LoggerF1')

function LoggerF1:__init(loggerConf)
  --[[
  loggerConf can have the following members
  logdir
  prefix
  transformOutputAndTarget
             (optional) a function for remapping the input and target
             can be useful e.g. for removing the ignored value
             or to remap if the network and targets are defined in terms of 
             different labels
  saveBest   if set, best networks are saved
  --]]
  self.logdir=loggerConf.logdir

  -- the epochLogger is used for writing information that is collected
  -- for each forward-backward and written once per epoch
  self.prefix=loggerConf.prefix or ""
  self.epochName=self.prefix.."Epoch.log" -- name of the per-epoch logger
  self.epochLogger=optim.Logger(paths.concat(self.logdir,self.epochName))
  self.epochLogger:setNames{'F1'} 
  self.epochLogger.showPlot = false

  self.bestF1=0
  
  self.transformOT=
    loggerConf.transformOutputAndTarget or function (o,t) return o,t end
  self.saveBest=loggerConf.saveBest

  self.nBins=1000
  self.hPos=torch.Tensor(self.nBins):zero()
  self.hNeg=torch.Tensor(self.nBins):zero()
end

function LoggerF1:addToLog(output,target)
  -- the input is ignored
  if output and target then 
    local o,t=self.transformOT(output,target)
    o=o:double()
    t=t:byte()
    local opos=o:maskedSelect(t:eq(2))
    local oneg=o:maskedSelect(t:eq(1))
    self.hPos:add(opos:histc(self.nBins,0,1))
    self.hNeg:add(oneg:histc(self.nBins,0,1))
  end
end

function LoggerF1:logEpochEnd(net)
  local positives=self.hPos:sum()
  local negatives=self.hNeg:sum()
  local truepositives=self.hPos:index(1,torch.linspace(self.nBins,1,self.nBins):long()):cumsum()
  local falsepositives=self.hNeg:index(1,torch.linspace(self.nBins,1,self.nBins):long()):cumsum()
  local predpositives=torch.add(truepositives,falsepositives)
  -- protext against zero division
  predpositives:maskedFill(predpositives:eq(0),1)
  local precision=torch.cdiv(truepositives,predpositives)
  local recall=torch.div(truepositives,positives)
  -- protext against zero division
  precision:maskedFill(precision:eq(0),1e-12)
  recall:maskedFill(recall:eq(0),1e-12)
  local suminv=torch.pow(precision,-1)+torch.pow(recall,-1)
  local f1s=torch.pow(suminv,-1):mul(2)
  local f1=f1s:max()
  self.epochLogger:add{f1}

  if net and self.saveBest then 
    if f1>self.bestF1 then
      self.bestF1=f1
      --net:clearState()
      torch.save(paths.concat(self.logdir,self.prefix..'_net_bestF1.t7'),net)
    end
  end
  self.hPos:zero()
  self.hNeg:zero()
end

function LoggerF1:logUpdate()
end
