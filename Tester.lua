
require "optim"
local Tester = torch.class('NetworkTraining.Tester')

--[[
--]]

function Tester:__init(testerConfig)
--[[
  testerConfig has the following fields
    dataProvider	an object that prepares input data and target outputs
		it implements the "DataProviderInterface"
    logger	the object for logging intermediate results; it implements the
                "LoggerInterface"
    bSize	the number of batches used for testing
    useCritForTest	optional; if present and evaluating to boolean true,
		a forward pass on the criterion is performed during testing,
		enabling logging the objective value for predictions on 
		test data
    sizeAverage optional; if present, for testing the criterion "sizeAverage"
                field is set to this value; only applicable if useCritForTest
                is set
    trainingMode	optional; if present, the network is used in training
		mode
--]]
  
  self.dataProvider=testerConfig.dataProvider
  self.logger=testerConfig.logger
--[[
  NetworkTraining.check_object_interface_compatibility
    (NetworkTraining.DataProviderInterface,self.dataProvider)
  NetworkTraining.check_object_interface_compatibility
    (NetworkTraining.LoggerInterface,self.logger)
--]]
  self.bSize=testerConfig.bSize 
  self.useCritForTest=testerConfig.useCritForTest
  self.sizeAverage=testerConfig.sizeAverage
  self.trainingMode=testerConfig.trainingMode
end

function Tester:test(net,crit)
--[[
  iterates over the test set, performs forward propagation
  and passes the results together with corresponding targets to logger
--]]
  if self.trainingMode then
    net:training()
  else
    net:evaluate()
  end
  local oldSA
  if crit and self.useCritForTest and self.sizeAverage~=nil then 
    -- for precise cost logging (batch size varies)
    oldSA=self.crit.sizeAverage
    crit.sizeAverage=false 
  end
  while self.dataProvider:getBatch(self.bSize) do
    local output=net:forward(self.dataProvider:getInput())
    local cost
    if self.useCritForTest and crit then
      cost=crit:forward(output,self.dataProvider:getTarget())
    else
      cost=nil
    end
    -- accumulate log information
    if self.logger then
      self.logger:addToLog(output,self.dataProvider:getTarget(),
                         self.dataProvider:getInput(),cost)
    end
  end
  -- compute final log information and write log
  if self.logger then
    self.logger:logEpochEnd(net)
  end
  if crit and self.useCritForTest and self.sizeAverage~=nil then
    crit.sizeAverage=oldSA
  end
  net:training()
end

