local nt=require "NetworkTraining"
local tester=torch.Tester()
local tests=torch.TestSuite()

function valFromIndex(k)
  return k
end

function lblFromIndex(k)
  return k+100
end

function prepareDataset(imsize,lblsize,nitems)
  is=torch.LongTensor(imsize)
  ls=torch.LongTensor(lblsize)
  is=torch.cat(torch.LongTensor{nitems},is)
  ls=torch.cat(torch.LongTensor{nitems},ls)
  img=torch.Tensor(is:storage())
  lbl=torch.Tensor(ls:storage()):long()
  for i=1,nitems do
    img[i]:fill(valFromIndex(i))
    lbl[i]:fill(lblFromIndex(i))
  end
  return img,lbl
end

local imgsize=torch.LongStorage{3,2,4}
local lblsize=torch.LongStorage{2,4}
local nitems=9
local bsize=4

local function augment(i,l)
  return i:add(2),l:add(-2)
end

function tests.testGetBatch()
  local img,lbl=prepareDataset(imgsize,lblsize,nitems)
  local ds=NetworkTraining.DatasetMemory(img:clone(),lbl:clone())
  local ignore={false,true}
  local totno
  for _,ign in pairs(ignore) do
    local dp=NetworkTraining.DataProviderDataset{
      dataset=ds,
      pfunc=augment,
      ignoreLast=ign
    }
    for ii=1,2 do
      totno=0
      while dp:getBatch(bsize) do
        inp=dp:getInput()
        tar=dp:getTarget()
        tester:eq(inp:size(1),tar:size(1))
        for k=1,inp:size(1) do
          totno=totno+1
          i=inp[k]
          l=tar[k]
          j=img[totno]:clone()
          m=lbl[totno]:clone()
          j,m=augment(j,m)
          tester:eq(i,j)  -- correct input
          tester:eq(l,m:double())
          tester:eq(i:size(),imgsize)  -- size correct
          tester:eq(l:size(),lblsize)
        end
      end
      if ign then
        tester:eq(totno,math.floor(nitems/bsize)*bsize)
      else
        tester:eq(totno,nitems)
      end
    end
  end
end

function isSlice(bigt1,slice1,bigt2,slice2)
  local b1,b2=augment(bigt1:clone(),bigt2:clone())
  for i=1,b1:size(1) do
    if torch.all(b1[i]:eq(slice1)) and 
       torch.all(b2[i]:eq(slice2)) then return true end
  end
  return false
end

function checkNotHashed(hashT,t)
  val=t:reshape(t:numel())[1]
  r=hashT[val]==nil
  hashT[val]=true
  return r
end

function tests.testGetBatchShuffle()
  local img,lbl=prepareDataset(imgsize,lblsize,nitems)
  local ds=NetworkTraining.DatasetMemory(img:clone(),lbl:clone())
  local bsize=4
  local ignore={false,true}
  local totno
  for _,ign in pairs(ignore) do
    local dp=NetworkTraining.DataProviderDataset{
      dataset=ds,
      ignoreLast=ign,
      pfunc=augment,
      shuffle=true
    }
    for ii=1,2 do
      local hashi={}
      local hashl={}
      totno=0
      while dp:getBatch(bsize) do
        inp=dp:getInput()
        tar=dp:getTarget()
        tester:eq(inp:size(1),tar:size(1))
        for k=1,inp:size(1) do
          totno=totno+1
          i=inp[k]
          l=tar[k]
          tester:eq(i:size(),imgsize)  --size correct
          tester:eq(l:size(),lblsize)
          tester:assert(isSlice(img,i,lbl:double(),l)) --proper input
          tester:assert(checkNotHashed(hashi,i)) --uniqueness
          tester:assert(checkNotHashed(hashl,l))
        end
      end
      if ign then
        tester:eq(totno,math.floor(nitems/bsize)*bsize)
      else
        tester:eq(totno,nitems)
      end
    end
  end
end

tester:add(tests)

tester:run()

