
--[[
	the code for network training is modular
	to make it easier to plug in new modules,
	I explicitly define the interfaces.
	This is a simple code for checking compatibility of a user-defined
	class with an interface.

	An interface is defined as a lua table.
	Each element of the table defines one required function.
	The keys are names of the functions,
	and the values are tables (again) describing the functions.

	The description should have the following fields:
	desc -	a string describing the role of the function,
		printed in case a member is missing

	Example interface with a single function
	myInterface = {
		["do_smth"]={
			["desc"]="a method for doing something"
		}
	}

	I make the description a table, in case it needs to be extended
	to contain more fields in the future.
--]]

function NetworkTraining.check_object_interface_compatibility(interface,object)
  t=torch.getmetatable(torch.type(object))
  for k,v in pairs(interface) do
    if t[k]==nil then
      error("\nthe object of class "..torch.type(object).." doesn't implement"..
            " method "..k..", ".."with the following functionality:\n"..v.desc)
    end
  end
end
