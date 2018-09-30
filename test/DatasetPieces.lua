
local nt=require "NetworkTraining"
local tester=torch.Tester()
local tests=torch.TestSuite()

function fillImgLbl(find,img,lbl)
  local dl=lbl:dim()
  local di=img:dim()
  assert(dl==di-1)
  for i=1,dl do assert(img:size(i+1)==lbl:size(i)) end
  img:zero()
  lbl:zero()
  local m=1
  for i=1,dl do
    for j=1,lbl:size(i) do
      img:narrow(i+1,j,1):add(j*m)
    end
    m=m*lbl:size(i)
  end
  img:add(m*find)
  lbl:copy(img[1])
end

function prepareDataset(imgdir,lbldir,files,imsize)
  os.execute("mkdir "..imgdir)
  os.execute("mkdir "..lbldir)
  local lblsize=torch.LongTensor(imsize):narrow(1,2,imsize:size()-1):clone():storage()
  vals={}
  function addToVals(e)
    assert(vals[e]==nil)
    vals[e]=true
  end
  for k,v in pairs(files) do
    local i=torch.Tensor(imsize):zero()
    local l=torch.Tensor(lblsize):zero()
    fillImgLbl(k,i,l)
    torch.save(paths.concat(imgdir,v),i)
    torch.save(paths.concat(lbldir,v),l)
    l:apply(addToVals)
  end
end

function removeDataset(imgdir,lbldir)
  os.execute("rm -rf "..imgdir) 
  os.execute("rm -rf "..lbldir)
end

local files={"file1","f2","f33","15145","asdg"}
local imgdir="ndkflh39t41_testImgDir"
local lbldir="paj90875gaf_testLblDir"
local imgsize=torch.LongStorage{3,5,6}
local lblsize=torch.LongStorage{5,6}

function testImgAndLabelRetrieval(patchesPerImage,patchSize,margin,imgsize,lblsize)
  prepareDataset(imgdir,lbldir,files,imgsize,lblsize)
  local ds=NetworkTraining.DatasetDisk(imgdir,lbldir,files)
  
  hash={}
  function check(v)
    if v~=0 then
      tester:assert(vals[v] == true)
      tester:assert(hash[v] == nil)
      hash[v]=true
    end
  end
  local dp=NetworkTraining.DatasetPieces(ds,patchesPerImage,patchSize,margin)
  for k=1,#files*patchesPerImage do
      i,l=dp:item(k)
      for j=1,i:size(1) do tester:eq(torch.cmul(i[j],l:ne(0):double()),l) end
      l:apply(check)
  end
  for k,v in pairs(vals) do tester:assert(hash[k]) end
  removeDataset(imgdir,lbldir) 
end

function tests:test2D()
  testImgAndLabelRetrieval( 4,torch.Tensor{4,5},torch.Tensor{1,2},imgsize,lblsize)
  testImgAndLabelRetrieval( 6,torch.Tensor{3,5},torch.Tensor{1,2},imgsize,lblsize)
  testImgAndLabelRetrieval( 6,torch.Tensor{3,5},torch.Tensor{1,1},imgsize,lblsize)
  testImgAndLabelRetrieval( 2,torch.Tensor{5,5},torch.Tensor{1,1},imgsize,lblsize)
  testImgAndLabelRetrieval( 1,torch.Tensor{5,6},torch.Tensor{1,1},imgsize,lblsize)
  testImgAndLabelRetrieval( 1,torch.Tensor{5,6},torch.Tensor{0,0},imgsize,lblsize)
  testImgAndLabelRetrieval( 9,torch.Tensor{2,2},torch.Tensor{0,0},imgsize,lblsize)
  testImgAndLabelRetrieval(12,torch.Tensor{3,3},torch.Tensor{1,1},imgsize,lblsize)
  testImgAndLabelRetrieval(30,torch.Tensor{1,1},torch.Tensor{0,0},imgsize,lblsize)
end

local imgsize3D=torch.LongStorage{3,5,6,7}
local lblsize3D=torch.LongStorage{5,6,7}

function tests:test3D()
  testImgAndLabelRetrieval( 1,torch.Tensor{5,6,7},torch.Tensor{1,1,1},imgsize3D,lblsize3D)
  testImgAndLabelRetrieval( 1,torch.Tensor{5,6,7},torch.Tensor{0,0,0},imgsize3D,lblsize3D)
  testImgAndLabelRetrieval(3*4*5,torch.Tensor{3,3,3},torch.Tensor{1,1,1},imgsize3D,lblsize3D)
  testImgAndLabelRetrieval(5*6*7,torch.Tensor{1,1,1},torch.Tensor{0,0,0},imgsize3D,lblsize3D)
  testImgAndLabelRetrieval(3*2*3,torch.Tensor{3,4,5},torch.Tensor{1,1,2},imgsize3D,lblsize3D)
  testImgAndLabelRetrieval(3*2*2,torch.Tensor{3,4,5},torch.Tensor{1,1,1},imgsize3D,lblsize3D)
  testImgAndLabelRetrieval(3*2*2,torch.Tensor{3,4,6},torch.Tensor{1,1,2},imgsize3D,lblsize3D)
  testImgAndLabelRetrieval(1*2*3,torch.Tensor{5,5,5},torch.Tensor{2,2,2},imgsize3D,lblsize3D)
  testImgAndLabelRetrieval(1*2*2,torch.Tensor{5,5,5},torch.Tensor{1,2,1},imgsize3D,lblsize3D)
  testImgAndLabelRetrieval(3*2*2,torch.Tensor{3,5,5},torch.Tensor{1,2,1},imgsize3D,lblsize3D)
end

tester:add(tests)

tester:run()

