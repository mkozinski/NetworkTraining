
require "optim"
local Tester = torch.class('NetworkTraining.TesterComposit')

--[[
--]]

function Tester:__init(testers)
--[[
    testers is a table of tester objects, presumably working on different data
--]]
  
  self.testers={}
  for k,v in pairs(testers) do
    table.insert(self.testers,v)
  end
  
end

function Tester:test(net,crit)
--[[
--]]
  for k,v in pairs(self.testers) do
    v:test(net,crit)
  end
end

