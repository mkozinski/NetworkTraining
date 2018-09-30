local nt=require "NetworkTraining"
local tester=torch.Tester()
local tests=torch.TestSuite()

function tests.InterfaceConformance()
  local i1 = {
    ["do_smth"]={
      ["desc"]="a method for doing something"
    }
  }
  local c1=torch.class("c1_testInterfaceConformance")
  function c1_testInterfaceConformance:do_smth() return 0 end
  local ci1=c1.new()
  f=function() nt.check_object_interface_compatibility(i1,ci1) end
  tester:assertNoError(f)
end

function tests.InterfaceNonconformance()
  local i1 = {
    ["do_smth"]={
      ["desc"]="a method for doing something"
    }
  }
  local c1=torch.class("c1_testInterfaceNonconformance")
  function c1_testInterfaceNonconformance:do_smth_else() return 0 end
  local ci1=c1.new()
  f=function() nt.check_object_interface_compatibility(i1,ci1) end
  tester:assertError(f)
end

tester:add(tests)

tester:run()

