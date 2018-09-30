
--[[
  this class manages forming batches of input data and ground truth annotations
  and counting epochs (cycles through the dataset)

  It implements the DataProviderInterface. 
  The interface is used by an object of class "Task" to control a data provider.

  Reading data is delegated to an object of class "Dataset",
  implementing a DatasetInterface.
--]]
--[[ 
  operation: 
  * initialize
  * repeatedly call getBatch to get batches;
    on epoch end the functions return false
  * after a call to getBatch, the input and target are retrieved by
    getInput and getTarget
  the 1st dim of input and target tensors produced by getBatch 
  corresponds to the index of the data in a batch
  
  if shuffle is used,
  the composition of batches is randomized (that is, different in each epoch)
  an epoch corresponds to traversing the whole dataset
  
  if ingoreLast is used
  the last batch is ignored if it is smaller than the requested size
--]]

local DataProviderDataset= 
  torch.class('NetworkTraining.DataProviderDataset')

function DataProviderDataset:__init(cfg)
--[[
  cfg can have the following fields:

  dataset is an object implementing the DatasetInterface
  used for data access (disk, memory, etc)
  it can also implement data augmentation mechanisms

  pfunc (optional argument) is a function used to process the batches of inputs
  and targets after they have been formed
  it can be used for example, for remapping the ground truth class indexes,
  performing batch-wise data augmentation, etc.
  one application is forwarding the data through GAN generator
  when training a GAN discriminator
  it is called as
  self.input,self.target=pfunc(self.input,self.target)
  where self.input and self.target are attributes of the calling object;
  it is expected to process the input and target in place,
  or process input and target, resize the input and target to fit the results,
  and copy the results into input and target.
  default is function(input,target) return input,target end 

  shuffle (optional) if set, the data items are acquired in random order

  ignoreLast (optional) if set, the last batch in an epoch is ignored,
  if it has size smaller than requested
--]]
                                     
  self.shuffle=cfg.shuffle
  self.ignoreLast=cfg.ignoreLast
  self.dataset= cfg.dataset
  -- randomly permuted list of training file indexes 
  if self.shuffle then
    self.inds=torch.randperm(self.dataset:noItems()) 
  else
    self.inds=torch.range(1,self.dataset:noItems())
  end
  self.in_ind=0   -- index of the last consumed training file
  self.input =torch.Tensor(0)
  self.target=torch.Tensor(0)
  self.pfunc = cfg.pfunc or function(input,target) return input,target end 
end

function DataProviderDataset:cuda()
  self.input =self.input :cuda()
  self.target=self.target:cuda()
  return self
end

function DataProviderDataset:getInput()
  return self.input
end

function DataProviderDataset:getTarget()
  return self.target
end

function DataProviderDataset:assertInputSizesEqual(sz1,sz2,i)
  assert(torch.all(torch.eq(torch.LongTensor(sz1),
                            torch.LongTensor(sz2,2,sz2:size()-1))), 
         " the input data item named "..
         self.dataset:inputName(self.inds[self.in_ind+i])..
         " has different size "..
         "than \'input\' (and some previously read input data)")
end

function DataProviderDataset:assertLabelSizesEqual(sz1,sz2,i)
  assert(torch.all(torch.eq(torch.LongTensor(sz1),
                            torch.LongTensor(sz2,2,sz2:size()-1))), 
         " the ground truth annotation named "..
         self.dataset:labelName(self.inds[self.in_ind+i])..
         " has different size "..
         "than \'target\' (and some previously read target data)")
end

function DataProviderDataset:getBatch(bSize)
  --[[
   loads a batch of input data into self.input 
   and a batch of ground truth annotations to self.target
   returns true on success and false if there is less than bSize data items
   left before the end of the epoch
   in the latter case self.input and self.target are not filled with data

   this function should to be called repeatetively to get consecutive batches
   after a successful call ('true' returned) the self.input and 
   self.target can be accessed by the getInput() and getTarget() methods
  --]]
  
  -- check for the end of epoch
  if bSize>self.dataset:noItems()-self.in_ind then
    if self.ignoreLast or self.in_ind>=self.dataset:noItems() then
      if self.shuffle then
        self.inds=torch.randperm(self.dataset:noItems())
      else
        self.inds=torch.range(1,self.dataset:noItems())
      end
      self.in_ind=0
      return false
    else
      bSize=self.dataset:noItems()-self.in_ind
    end
  end
  -- first, resize the self.input and self.target
  -- to match the expected batch size
  -- this requires checking the size of the first data item
  local im,lb=self.dataset:item(self.inds[self.in_ind+1])

  local szin=NetworkTraining.appendNum2Storage(im:size(),bSize)
  self.input:resize(szin)
  self.input[1]:copy(im)
  local szta
  if lb then
    -- it is possible that the dataset does not have labels
    -- but if it has, then check the expected size of the target batch
    szta=NetworkTraining.appendNum2Storage(lb:size(),bSize)
    self.target:resize(szta)
    self.target[1]:copy(lb)
  end
  local imsz
  local lbsz
  -- copy the remaining data items
  for i_f=2,bSize do
    im,lb=self.dataset:item(self.inds[self.in_ind+i_f])
    imsz=im:size()
    self:assertInputSizesEqual(imsz,szin,i_f)
    self.input[i_f]:copy(im)
    if lb then -- the labels can be absent
      lbsz=lb:size()
      self:assertLabelSizesEqual(lbsz,szta,i_f)
      self.target[i_f]:copy(lb)
    end
  end

  self.input,self.target=self.pfunc(self.input,self.target)
  self.in_ind=self.in_ind+bSize

  return true
end

