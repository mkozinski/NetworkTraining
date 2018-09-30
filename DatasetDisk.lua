--[[
  a dataset class 
  handles reading data stored in individual torch files on the disk
  one tensor per file; works for arbitrary size tensors

  the augmentTrain and augmentTest are optional arguments
  both are functions that are called in the following way
  input,label=proprocXXX(input,label)
  and are expected to apply preprocessing to the input data and labels
--]]
require "image"

local Dataset=torch.class('NetworkTraining.DatasetDisk')

function Dataset:__init(imgDir,lblDir,files,augment, opts)
  self.imgDir=imgDir
  self.lblDir=lblDir
  self.files =files
  self.augment=augment or function (i,t) return i,t end
  self.imgPrefix= opts and opts.imgPrefix  or ""
  self.imgPostfix=opts and opts.imgPostfix or ""
  self.lblPrefix= opts and opts.lblPrefix  or ""
  self.lblPostfix=opts and opts.lblPostfix or ""
  self.inputType =opts and opts.inputType  or "tensor" -- or "image"
  self.labelType =opts and opts.labelType  or "tensor" -- or "image"
  self.inputChnum=opts and opts.inputChnum or 3
end

function Dataset:noItems()
  return #self.files
end

function Dataset:readImgLbl(fname)
  local ffname=paths.concat(self.imgDir,self.imgPrefix..fname..self.imgPostfix)
  local im
  if self.inputType=="image" then
    im=image.load(ffname,self.inputChnum,"float")
  else
    im=torch.load(ffname)
  end
  local lb
  if self.lblDir then
    ffname=paths.concat(self.lblDir,self.lblPrefix..fname..self.lblPostfix)
    if self.labelType=="image" then
      lb=image.load(ffname,1,"float"):squeeze()
    else
      lb=torch.load(ffname)
    end
  end

  return im,lb
end

function Dataset:item(ind) 
  local img,lbl =  self:readImgLbl(self.files[ind])
  return self.augment(img,lbl)
end

function Dataset:inputName(ind)
  return paths.concat(self.imgDir,self.files[ind])
end

function Dataset:labelName(ind)
  local r=""
  if self.lblDir then r=paths.concat(self.lblDir,self.files[ind]) end
  return r
end
