--[[
  a dataset class 
  handles reading data stored in a single big tensor in memory

  "augment" is an optional argument
  it is a function called in the following way
  input,label=augment(input,label)
  and it is expected to apply preprocessing to the input data and labels
--]]

local DatasetMemory, parent =torch.class('NetworkTraining.DatasetMemory')

function DatasetMemory:__init(imgs,lbls,augment)
  self.img=imgs
  self.lbl=lbls
  self.augment=augment or function (i,t) return i,t end
end

function DatasetMemory:noItems()
  return self.img:size(1)
end

function DatasetMemory:item(ind) 
  local img=self.img[ind]
  local lbl=self.lbl and self.lbl[ind]
  return self.augment(img:clone(),lbl:clone())
end

function DatasetMemory:inputName(ind)
  return "image number "..tonumber(ind)
end

function DatasetMemory:labelName(ind)
  return "ground truth number "..tonumber(ind)
end


