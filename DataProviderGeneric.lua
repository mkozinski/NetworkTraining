
--[[
  i use this data provider when i need random input
  but it can be used in many ways, by specifying "rfunc" accordingly
  it can handle inputs and targets in form of tables
--]]

local DataProviderGeneric= 
  torch.class('NetworkTraining.DataProviderGeneric')

function DataProviderGeneric:__init(cfg) 
--[[
  cfg can have the following fields:

  noItems    the number of items in the dataset
  rfunc      a function generating the target and input of requested size
             called as rfunc(size)

  ignoreLast (optional) if set, the last batch in an epoch is ignored,
             if it has size smaller than requested
--]]

  self.noItems= cfg.noItems
  self.in_ind=0   -- index of the last consumed training file
  self.ignoreLast=cfg.ignoreLast
  self.rfunc=cfg.rfunc
end

function DataProviderGeneric:getInput()
  return self.input
end

function DataProviderGeneric:getTarget()
  return self.target
end

function DataProviderGeneric:getBatch(bSize)
  -- check for the end of epoch
  if bSize>self.noItems-self.in_ind then
    if self.ignoreLast or self.in_ind>=self.noItems then
      self.in_ind=0
      return false
    else
      bSize=self.noItems-self.in_ind
    end
  end
  self.input,self.target=self.rfunc(bSize)
  self.in_ind=self.in_ind+bSize
  return true
end

