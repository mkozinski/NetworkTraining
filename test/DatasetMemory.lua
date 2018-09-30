
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
local function augment(i,l)
  return i:add(2),l:add(-2)
end

function tests.testImgAndLabelRetrieval()
  local img,lbl=prepareDataset(imgsize,lblsize,9)
  local ds=NetworkTraining.DatasetMemory(img:clone(),lbl:clone(),augment)
  for ii=1,2 do
    for k=1,img:size(1) do
      i,l=ds:item(k)
      j=img[k]:clone()
      m=lbl[k]:clone()
      j,m=augment(j,m)
      tester:eq(i,j)
      tester:eq(l,m)
      tester:eq(i:size(),imgsize)
      tester:eq(l:size(),lblsize)
    end
  end
end

function tests.testNameRetrieval()
  img,lbl=prepareDataset(imgsize,lblsize,11)
  local ds=NetworkTraining.DatasetMemory(img,lbl,augment)
  for ii=1,2 do
    for k=1,img:size(1) do
      i=ds:inputName(k)
      l=ds:labelName(k)
      j="image number "..k
      m="ground truth number "..k
      tester:eq(i,j)
      tester:eq(l,m)
    end
  end
end

function tests.testNoItems()
  local ni=123
  img,lbl=prepareDataset(imgsize,lblsize,ni)
  local ds=NetworkTraining.DatasetMemory(img,lbl)
  tester:eq(ni,ds:noItems())
end

tester:add(tests)

tester:run()

