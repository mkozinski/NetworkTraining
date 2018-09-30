require "optim"
local Task = torch.class('NetworkTraining.Task')

--[[
  this class models a single task in a (possibly) multi-task learning scenario
  it represents the trained network, the data providers, the cost function, 
  the optimization engine and the loggers;

  it implements the "TaskInterface" that is used by the "Setup"
  to control the tasks
  
  it controls the Logger, Tester and DataProvider by the corresponding 
  interfaces
--]]

function Task:__init(taskConfig)
--[[
  taskConfig has the following fields
    net		the network forward and backward are called on this object
    netParts	optional; a table of sub-networks of net;
		to be used when some but not all parameters of net are shared,
		for example when training an autoencoder
		and sharing the encoder with an image classification network;
                see below for more explanations
    crit	the criterion; forward and backward is called on this object
		criterions from the torch nn package are normally used here
    optimState	a table of tables of parameters of the optimization algorithm
		algorithms from the torch optim package normally require 
                some parameters to be specified. optimState should be a table
                of parameter tables, one for each netPart;
                of netParts are not used, optimState should be a one element
                table, containing the table of parameters
		elements of optimState are passed to the optimization function
		specified in the optimEngine parameter,
		for example the "optim.sgd" from torch optim package
    dataProvider	an object that prepares input data and target outputs
		it implements the "DataProviderInterface"
    logger	the object for logging intermediate results; it implements the
                "LoggerInterface"
    bSize	the number of batches used for training
    bSizeTest	optional; the number of batches used for testing;
		default is bSize
    n_fbs	optional; the number of forward-backward propagations
		before updates values greater than one can be used
                to get more stable gradients with respect to weights,
                in cases where the batch size is limited by memory;
                default is 1   
    test_every	optional; if defined, the task is tested every "test_every"
                epochs;
    optimEngine	optional; the optimization engine
		(implementing, e.g., momentum and weight decay)
		i use functions from the torch optim package here;
		note- it this is the function to which elements of optimState
		are passed;
		defualt is optim.sgd
    useCritForTest	optional; if present and evaluating to boolean true,
		a forward pass on the criterion is performed during testing,
		enabling logging the objective value for predictions on 
		test data
    tester      optional; a tester object implementing the TesterInterface
--]]
  
--[[ the idea of "netParts" is that if you are sharing params,
     you should share all parameters of a single "netPart"

     imagine a network with consecutive layers A and B,
     and another one with consecutive layers A and C.
     suppose you want to share the weights of layers named A in both nets
     if you call getParameters() on the whole 1st network, containing AB,
     the weights of layers A and B will be represented in a *contiguous* 
     chunk of memory, and the tensors in the network will be set to
     "point to" parts of this contiguous chunk
     A subsequent call to getParameters() on the 2nd net, containing AC,
     will create anothe contiguous chunk, for storing weights of A and C,
     and set the the tensors of A and C to point there.
     This breaks parameter sharing; 
     Putting it differently, it is impossible to have both
     A and C, and A and B to have their weights stored in a contiguous 
     chunk of mem.

     Summarizing, breaking the network into "netParts" is aimed at enabling
     sharing only a subset of network parameters while still using 
     the getParameters() function
--]]
  self.net=taskConfig.net
  self.netParts=taskConfig.netParts or {self.net}
  self.crit=taskConfig.crit
  self.optimState=taskConfig.optimState
  self.dataProvider=taskConfig.dataProvider
  NetworkTraining.check_object_interface_compatibility
    (NetworkTraining.DataProviderInterface,self.dataProvider)
  self.logger=taskConfig.logger
  NetworkTraining.check_object_interface_compatibility
    (NetworkTraining.LoggerInterface,self.logger)
  self.bSize=taskConfig.bSize
  self.bSizeTest=taskConfig.bSizeTest or self.bSize
  self.n_fbs=taskConfig.n_fbs or 1
  self.test_every=taskConfig.test_every 
  self.optimEngine=taskConfig.optimEngine or optim.sgd
  self.useCritForTest=taskConfig.useCritForTest
  self.params={}
  self.gradParams={}
  for i_part=1,#self.netParts do 
  --[[ attention, this flattens the parameters!
       problems may result if you share some but not all 
       parameters of a single "netPart" --]]
    self.params[i_part],self.gradParams[i_part]=
      self.netParts[i_part]:getParameters()
  end
  self.no_epochs=0 -- number of epochs passed
  self.n_iter=0    -- number of forward-backward passes
  self.no_updates=0-- number of weight updates performed
  
  self.net:zeroGradParameters()

  if taskConfig.tester then
    NetworkTraining.check_object_interface_compatibility
      (NetworkTraining.TesterInterface,taskConfig.tester)
    self.tester=taskConfig.tester
  end
end

function Task:perform()
--[[
   this function performs forward-backward on the network n_fbs times
   this results in gradient with respect to parameters being accumulated
   in the 'net' object
--]]
  for ifb=1,self.n_fbs do
    --local t=torch.tic()
    if not self.dataProvider:getBatch(self.bSize) then
    --[[ an epoch has ended
         this can happen during accumulation of the gradient and before update
         (when self.n_fbs>1)
         however, this is not a problem, since self:test() does not clear
         the gradients --]]
      self.logger:logEpochEnd() -- compute accuracies and write to log files
      self.no_epochs=self.no_epochs+1
      if self.tester and self.test_every and 
         self.no_epochs % self.test_every == 0 then
        -- only test once every self.test_very epochs
        self.tester:test(self.net,self.crit)
      end
      assert(self.dataProvider:getBatch(self.bSize),
             "dataProvider:getBatch() returned false two times in a row; "..
             "too large batch size requested ?") --prepare a new batch
    end
    --print("time spent reading in data",torch.toc(t))
    --t=torch.tic()
    local output=self.net :forward(self.dataProvider:getInput())
    local cost  =self.crit:forward(output,self.dataProvider:getTarget())
    local dfdx  =self.crit:backward(output,self.dataProvider:getTarget())
    self.net:backward(self.dataProvider:getInput(),dfdx)
    self.logger:addToLog(output,self.dataProvider:getTarget(),
                         self.dataProvider:getInput(),cost,dfdx)
    -- accumulate logging information
    self.n_iter=self.n_iter+1
    --print("forward, backward and logging time",torch.toc(t))
  end
end

function Task:update()
  --[[
   this function performs an update to network weights,
   on a part-by-part basis 
   the gradients are already stored in the "net" member
   in result of a call to perform()
   after the update the gradients are zeroed
  --]]
  for i_parts=1,#self.netParts do
    local f=function() return 0,self.gradParams[i_parts] end
    if not self.optimState[i_parts] then 
      -- forgetting to define the optimState may go unnoticed...
      -- and the default parameters are not always appropriate
      print('warning no optimState[',i_parts,'] defined');
    end
    self.optimEngine(f,self.params[i_parts],self.optimState[i_parts])
  end
  self.no_updates=self.no_updates+1
  -- save the network, the parameter gradients,etc
  self.logger:logUpdate(self.net,self.optimState,self.params,self.gradParams)
  -- zero the gradients after they have been logged
  for i_parts=1,#self.netParts do
    self.netParts[i_parts]:zeroGradParameters()
  end
end

function Task:clipParams(c)
  --[[
    clip the weights of the network so that their absoluve value
    does not exceed "c"
  --]]
  for k,m in pairs(self.net:listModules()) do
    if m.weight then
      m.weight:clamp(-c,c)
    end
  end
end
