
--[[ 
  implements the LoggerInterface
  this logger implements the basic logging functionality,
  that is common to many problems
  the logged values are based on:
  objective value and gradient
  parameter value and gradient
  the logger also saves the network during training


  it creates three log files
  basicUpdate.log  updated every time a network update is performed.
                   stores parameters norms and norms of parameter gradients
  basicEpoch.log   updated every time a training epoch is finished
                   stores: average objective value (over epoch),
                   average objective gradient norm, 
                   average maximum per-pixel gradient
  basicTest.log    updated every time a test is performed.
                   stores average objective value
--]]

local LoggerBasic= 
  torch.class('NetworkTraining.LoggerBasic')

function LoggerBasic:__init(loggerConf)
  --[[
  loggerConf can have the following members
  logdir
  nParams    number of sub-networks into which the network is divided
             (see the documentation of "Task" for details)
  noData     (optional) the total number of data items
  bSize     (optional) the batch size used for training
  saveEvery  (optional) if defined, the network is saved every 'saveEvery' 
             updates in the logdir, under the name 'net_last.t7'
             the previous network is overwritten on ech save
  saveE2     (optional) if you want to save intermediate network states,
             set saveE2 to the number of updates between consecutive saves
             in file 'netBestCost.t7'
  logWeight  (optional) if set to true, the norm of parameter vector
             is logged in basicUpdate.log
  logWeightGrad (optional) if set to true, the norm of parameter gradient
             is logged in basicUpdate.log
  logAvgGrads (optional) if set to true the average gradient of the objective
             with respect to its input is logged in basicEpoch.log
  logMaxGrads (optional) if set to true, the average maximum (w.r.t. pixels)
             gradient of the objective with respect to its input is logged
             in basicEpoch.log
  prefix     (optional) the prefix to log file names;
             two files are created <prefix>Epoch.log and <prefix>Update.log
  saveBest   (optional) if set, the best network is saved
  saveBatchEvery (optional) if set, the logger will save the first batch
             of inputs targets and outputs in an epoch every saveBatchEvery
             epochs; this data is ovewritten at each epoch
  saveBatchE2 (optional) if set, the logger will save the first batch
             of inputs targets and outputs in an epoch every saveBatchE2
             epochs; this data is not ovewritten 
  saveAllBatchesEvery (optional) if set, the logger will save all batches
             every "saveEachBatchEvery" epochs; this data is not overwritten
  transformOutputAndTarget
             (optional) a function for remapping the input and target
             and output and gradient
             can be useful e.g. for removing the ignored value
             or to remap if the network and targets are defined in terms of 
             different labels
             signature is transformOutputAndTarget(output,target,input,gradient)
  clearState a binary variable; if set to true, the network will be cleared
             before it is damped to disk; otherwise, it is dumped to disk
             with all the feature maps, generating large files (often in GBs!); 
             the default is not to clear the network and to dump large files,
             because calling "clearState" on a network and re-creating the 
             feature tensors after that increases memory consumption(!)
  --]]
  self.logdir=loggerConf.logdir
  self.totalCost=0    -- approximates the loss on the training set
                      -- by summing costs for individual batches
  self.norm_dfdx=0    -- approximates the norm of the gradient of the obejctive
  self.max_dfdx=0     -- average (over pixels) maximum gradient of the objective
  self.n_epochs=0     -- epoch counter
  self.i_batch=0      -- batch index in the epoch

  self.prefix=loggerConf.prefix or ""
  -- the epochLogger is used for writing information that is collected
  -- for each forward-backward and written once per epoch
  self.epochName=self.prefix.."Epoch.log"
  self.epochLogger=optim.Logger(paths.concat(self.logdir,self.epochName))
  self.logWeight    =loggerConf.logWeight
  self.logWeightGrad=loggerConf.logWeightGrad
  self.logAvgGrads  =loggerConf.logAvgGrads
  self.logMaxGrads  =loggerConf.logMaxGrads
  local names={'objective'}
  if self.logAvgGrads then table.insert(names, 'norm of objective gradient')  end
  if self.logMaxGrads then 
    table.add(names, 'average maximum per-pixel gradient norm')
  end
  self.epochLogger:setNames(names)
  names=nil
  self.epochLogger.showPlot = false

  self.updateName=self.prefix.."Update.log"
  -- the update logger writes information once per weights update
  -- and has a pair of columns for each "netPart"
  self.updateLogger=optim.Logger(paths.concat(self.logdir,self.updateName))
  local lnames={}
  if loggerConf.nParams then
    for ip=1,loggerConf.nParams do
      if self.logWeight then table.insert(lnames,'param '..ip..' norm ') end
      if self.logWeightGrad then table.insert(lnames,'grad '..ip..' norm ') end
    end
  else
    if self.logWeight then table.insert(lnames,'param norm ') end
    if self.logWeightGrad then table.insert(lnames,'grad norm ') end
  end
  self.updateLogger:setNames(lnames)
  self.updateLogger.showPlot = false

  self.saveNetEvery=loggerConf.saveEvery
  self.saveNextNetEvery=loggerConf.saveE2
  
  -- for normalizing all the parameters that get summed over an epoch
  -- logFac is equal to number of batches in an epoch
  if loggerConf.noData and loggerConf.bSize then 
    self.logFac=math.floor(loggerConf.noData/loggerConf.bSize)
  else
    self.logFac=1 
  end
 
  self.saveBest=loggerConf.saveBest
  self.bestCost=math.huge
  self.nUpdates=0
 
  self.saveBatchEvery = loggerConf.saveBatchEvery
  self.saveBatchE2    = loggerConf.saveBatchE2
  self.saveAllBatchesEvery=loggerConf.saveAllBatchesEvery
  self.transformOT=
    loggerConf.transformOutputAndTarget or function (o,t,i,d) return o,t,i,d end
end

function LoggerBasic:addToLog(output,target,input,f,dfdx)
  local o,t,i,g=self.transformOT(output,target,input,dfdx)
  if f then self.totalCost=self.totalCost+f/self.logFac end
  if g and self.logAvgGrads then 
    self.norm_dfdx=self.norm_dfdx+g:norm()/self.logFac 
  end
  if g and self.logMaxGrads then
    self.max_dfdx=self.max_dfdx+g:norm(2,2):max()/self.logFac
  end
  local lp=paths.concat(self.logdir,self.prefix)
  local ne=self.n_epochs
  local ib=self.i_batch
  if self.i_batch==0 then -- first batch of the epoch
    if self.saveBatchEvery and self.n_epochs % self.saveBatchEvery == 0 then
      torch.save(paths.concat(lp.."_input_last.t7"), i:double())
      torch.save(paths.concat(lp.."_target_last.t7"),t:double())
      torch.save(paths.concat(lp.."_output_last.t7"),o:double())
    end
    if self.saveBatchE2 and self.n_epochs % self.saveBatchE2 == 0 then
      torch.save(paths.concat(lp.."_input_" ..ne..".t7"),i:double())
      torch.save(paths.concat(lp.."_target_"..ne..".t7"),t:double())
      torch.save(paths.concat(lp.."_output_"..ne..".t7"),o:double())
    end
  end
  if self.saveAllBatchesEvery and self.n_epochs % self.saveAllBatchesEvery == 0 then
    torch.save(paths.concat(lp.."_input_" ..ne.."_"..ib..".t7"),i:double())
    torch.save(paths.concat(lp.."_target_"..ne.."_"..ib..".t7"),t:double())
    torch.save(paths.concat(lp.."_output_"..ne.."_"..ib..".t7"),o:double())
  end
  self.i_batch=self.i_batch+1
end

function LoggerBasic:logEpochEnd(net)
  local lt={self.totalCost}
  if self.logAvgGrads then table.insert(lt,self.norm_dfdx) end
  if self.logMaxGrads then table.insert(lt,self.max_dfdx)  end
  self.epochLogger:add(lt)
  self.totalCost=0
  self.norm_dfdx=0
  self.avg_max_grad=0
  if self.saveBest and self.totalTestCost<self.bestCost then
    self.bestCost=self.totalTestCost
    torch.save(paths.concat(self.logdir,self.prefix.."_net_BestCost.t7"),net)
  end
  self.n_epochs=self.n_epochs+1
  self.i_batch=0
end

function LoggerBasic:logUpdate(net,optimState,params,gradParams)
  local logTab={}
  for i=1,#params do 
    if self.logWeight then table.insert(logTab,params[i]:norm()) end
    if self.logWeightGrad then table.insert(logTab,gradParams[i]:norm()) end
  end
  self.updateLogger:add(logTab)
  self.nUpdates=self.nUpdates+1
  local lp=paths.concat(self.logdir,self.prefix)
  local ne=self.n_epochs
  if self.saveNetEvery and self.nUpdates % self.saveNetEvery == 0 then
    if self.clearState then
      net:clearState()
    end
    torch.save(lp..'_net_last.t7',net)
    torch.save(lp..'_os_last.t7',optimState)
  end
  if self.saveNextNetEvery and self.nUpdates % self.saveNextNetEvery == 0 then
    net:clearState() -- it is better to clear always when dumping many files
    torch.save(lp..'_netUpdate'..ne..'.t7',net)
    torch.save(lp..'_osUpdate' ..ne..'.t7',optimState)
  end
end

