--[[
  the application of this class is mainly for testing
  when a test image cannot be fed into a convolutional network
  in its original size, due to memory constraints

  such images/volumes is cut into patches with overlapping margins, 
  that should match the radius of the receptive field of the network
  within the margins, the ground truths contain class index "0"
  indicating that these regions should be ignored during evaluation
  this matches the convention of the torch optim.ConfusionMatrix class

  testing a network on data from this dataset gives the same results
  as testing it on the original data,
  but does not require processing large images

  any other dataset can be wrapped with this piece-wise functionality
--]]

local Dataset=torch.class('NetworkTraining.DatasetPieces')

function Dataset:__init(dataset,
  nChunksPerFile,size,margin,lblHasChannelDim)

  self.dataset=dataset
  self.sz=size -- vector; size of patches to crop
  self.msz=margin  -- vector; size of the margins
  self.npf=nChunksPerFile -- how many patches in a single file
  self.last=0  -- the index of the last read image
  self.lblCh=lblHasChannelDim -- normally labels have no channel dimension
end

function Dataset:item(ind)
  assert(ind<=self:noItems(),"requested  crop index "..ind.." but i only have "..self:noItems().." crops") 
  local imgind=math.ceil(ind/self.npf) -- which image to read
  if not self.img or not self.last or self.last~=imgind then --read img
    self.img,self.lbl=self.dataset:item(imgind)
    self.last=imgind
  end
  --how many patches in the input, number of patches per dimension, 
  --step in image coordinates between neighboring patches
  self.isz=torch.LongTensor(self.img:size()):double()
  self.is2=self.isz:narrow(1,2,self.isz:numel()-1):clone() --img spatial size
  self.step=self.sz-2*self.msz -- the stride between two neighboring crops in image coordinates
  self.nppd=torch.cdiv(torch.add(self.is2,-2,self.msz),self.step):ceil()  --num patches per dimension
  self.npps=self.nppd:clone():fill(1) -- num patches per unit index step in each dimension
  for i=self.nppd:numel()-1,1,-1 do
    self.npps[i]=self.npps[i+1]*self.nppd[i+1]
  end
  --patch index within image in "patch coordinates"
  local pi=self.is2:clone():fill(1) --the index of the patch to be retrieved along each input dim ("2nd patch along 3rd dimension")
  local patchind=ind-(imgind-1)*self.npf
  local res=patchind-1 -- "residual"
  for i=1,pi:numel() do
    pi[i]=pi[i]+math.floor(res/self.npps[i])
    res=res-(pi[i]-1)*self.npps[i]
  end
  assert(res==0)

  -- get the image coordinates of the image and lbl crops to be retrieved
  -- and the coordinates within the space of the output patches
  -- (sometimes the copied data does not fill the patch completely)
  local srcindimg={{}} -- what to copy from input data --copy all channels
  local dstindimg={{}} -- where to copy the input crop within the patch
  local srcindlbl={}   -- what to copy friom the label
  local dstindlbl={}   -- where to copy the label crop within the label patch
  if self.lblCh then table.insert(srcindlbl,{}) table.insert(dstindlbl,{}) end
  for d=1,self.img:dim()-1 do
    if self.is2[d]<self.sz[d] then  -- requested patch larger than input
      local szd=(self.sz[d]-self.is2[d])
      local m1=math.floor(szd/2)
      table.insert(srcindimg,{1,self.is2[d]})
      table.insert(dstindimg,{m1+1,self.is2[d]})
      table.insert(srcindlbl,{1,self.is2[d]})
      table.insert(dstindlbl,{m1+1,self.is2[d]})
    else
      local sm=self.msz[d] -- size of "leftmost" margin
      local sz=self.sz[d]-2*self.msz[d] -- size of label contents to be copied
      if pi[d]==1 then -- first patch along dimension d=>no leftmost margin
        sm=0
        sz=sz+self.msz[d]
      end
      local si=1+(pi[d]-1)*(self.sz[d]-2*self.msz[d]) -- start index of the block to be copied from the input data
      if si>=self.is2[d]-self.sz[d]+1 then -- last patch along dimension d=> should end at input border,
        sm=sm+si-self.is2[d]+self.sz[d]-1 -- larger margin on the left
        si=self.is2[d]-self.sz[d]+1  -- shift patch left so that it ends at input border
        sz=self.sz[d]-sm -- larger margin on the left (because patch shifted)
      end
      table.insert(srcindimg,{si,si+self.sz[d]-1})
      table.insert(dstindimg,{1,self.sz[d]})
      table.insert(srcindlbl,{si+sm,si+sm+sz-1})
      table.insert(dstindlbl,{sm+1,sm+sz})
    end
  end
  self.isz[{{2,self.isz:numel()}}]:copy(self.sz) --size of output crop
  local oimg=self.img.new(self.isz:long():storage()):zero()
  oimg[dstindimg]:copy(self.img[srcindimg])
  local olbl
  if self.lbl then
    if self.lblCh then 
      local sz1=torch.Tensor(self.sz:numel()+1)
      sz1[1]=self.lbl:size(1)
      sz1:narrow(1,2,self.sz:numel()):copy(self.sz)
      olbl=self.lbl.new(sz1:long():storage()):zero()
    else
      olbl=self.lbl.new(self.sz:long():storage()):zero()
    end
    --print("\ndstindLbl",dstindlbl,"\nolbl:size()",olbl:size(),"\nsrcindLbl",srcindlbl,"\nself.lbl:size()",self.lbl:size())
    olbl[dstindlbl]:copy(self.lbl[srcindlbl])
  end
  return oimg,olbl
end

function Dataset:noItems() return self.dataset:noItems()*self.npf end

function Dataset:inputName(ind) 
  return self.dataset:inputName(math.ceil(ind/self.npf)) 
end

function Dataset:labelName(ind) 
  return self.dataset:labelName(math.ceil(ind/self.npf)) 
end

