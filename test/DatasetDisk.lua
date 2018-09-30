local nt=require "NetworkTraining"
local tester=torch.Tester()
local tests=torch.TestSuite()

function valFromIndex(k)
  return k
end

function lblFromIndex(k)
  return k+100
end

function prepareDataset(imgdir,lbldir,files,imsize,lblsize)
  os.execute("mkdir "..imgdir)
  os.execute("mkdir "..lbldir)
  for k,v in pairs(files) do
    local i=torch.Tensor(imsize):fill(valFromIndex(k))
    local l=torch.Tensor(lblsize):fill(lblFromIndex(k))
    torch.save(paths.concat(imgdir,v),i)
    torch.save(paths.concat(lbldir,v),l:long())
  end
end

function removeDataset(imgdir,lbldir)
  os.execute("rm -rf "..imgdir) 
  os.execute("rm -rf "..lbldir)
end

local files={"file1","f2","f33","15145","asdg"}
local imgdir="ndkflh39t41_testImgDir"
local lbldir="paj90875gaf_testLblDir"
local imgsize=torch.LongStorage{3,2,4}
local lblsize=torch.LongStorage{2,4}
local function augment(i,l)
  return i:add(2),l:add(-2)
end

function tests.testImgAndLabelRetrieval()
  prepareDataset(imgdir,lbldir,files,imgsize,lblsize)
  local ds=NetworkTraining.DatasetDisk(imgdir,lbldir,files,augment)
  for ii=1,2 do
    for k,v in pairs(files) do
      i,l=ds:item(k)
      j=torch.load(paths.concat(imgdir,v))
      m=torch.load(paths.concat(lbldir,v))
      j,m=augment(j,m)
      tester:eq(i,j)
      tester:eq(l,m)
      tester:eq(i:size(),imgsize)
      tester:eq(l:size(),lblsize)
    end
  end
  removeDataset(imgdir,lbldir) 
end

function tests.testNameRetrieval()
  prepareDataset(imgdir,lbldir,files,imgsize,lblsize)
  local ds=NetworkTraining.DatasetDisk(imgdir,lbldir,files,augment)
  for ii=1,2 do
    for k,v in pairs(files) do
      i=ds:inputName(k)
      l=ds:labelName(k)
      j=paths.concat(imgdir,v)
      m=paths.concat(lbldir,v)
      tester:eq(i,j)
      tester:eq(l,m)
    end
  end
  removeDataset(imgdir,lbldir) 
end

function tests.testNoItems()
  prepareDataset(imgdir,lbldir,files,imgsize,lblsize)
  local ds=NetworkTraining.DatasetDisk(imgdir,lbldir,files,augment)
  tester:eq(#files,ds:noItems())
  removeDataset(imgdir,lbldir)
end

tester:add(tests)

tester:run()

