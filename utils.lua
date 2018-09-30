require "image"

function NetworkTraining.appendNum2Storage(s,n)
-- append the number n in the beginning of a LongStorage
  local s2=torch.LongStorage(s:size()+1)
  local s3=torch.LongStorage(s2,2)
  s3:copy(s)
  s2[1]=n

  return s2
end


function NetworkTraining.class2color(ci,map)
--[[
  produce a colorful class map, from a class index tensor
  ci - class indexes; two last dimensions are image width and height
       possible dimensionality
       2d - widthXheight
       3d - batchXwidthXheight
       4d - batchX1XwidthXheight
  map - color map: a table, indexes are class indexes, values are tensors
        encoding colors
  note: the class indexes need to be between 1 and some num N
--]]
  local maxind=ci:max()
  assert(maxind<=#map,'more classes than map entries')
  local cie,b,o,outr,outg,outb
  if ci:dim()==2 then
    -- height x width
    o=torch.Tensor(3,ci:size(1),ci:size(2)):zero()
    outr=o:narrow(1,1,1)
    outg=o[2]
    outb=o[3]
  elseif ci:dim()==3 or ci:dim()==4 then
    -- batch x height x width or batch x 1 x height x width
    if ci:dim()==4 then
      -- batch x 1 x height x width
      assert(ci:size(2)==1,"unitary second dim expected")
      ci=ci:reshape(ci:size(1),ci:size(3),ci:size(4))
    end
    o=torch.Tensor(ci:size(1),3,ci:size(2),ci:size(3)):zero()
    outr=o:narrow(2,1,1)
    outg=o:narrow(2,2,1)
    outb=o:narrow(2,3,1)
  else
    error('ci needs to have 2, 3 or 4 dimensions')
  end

  for i,c in pairs(map) do
    local b=ci:eq(i):double()
    outr:add(outr,torch.mul(b,c[1]))
    outg:add(outg,torch.mul(b,c[2]))
    outb:add(outb,torch.mul(b,c[3]))
  end
  return o
end

function NetworkTraining.jitterImgLbl(img,lbl,marginV,marginH)
  if marginH==nil then marginH=marginV end
  if marginV>0 or marginH>0 then
    local vOff=math.floor(math.random()*marginV)
    local hOff=math.floor(math.random()*marginH)
    local h,w=img:size(2),img:size(3)
    img=image.crop(img,hOff,vOff,w-marginH+hOff,h-marginV+vOff)
    lbl=image.crop(lbl,hOff,vOff,w-marginH+hOff,h-marginV+vOff)
  end
  return img,lbl
end

function NetworkTraining.jitterImgLbl2(img,lbl,h,w)
  errmsg=" requested height is "..h.." requested width is "..w..
         " image height is "..img:size(2).." image width is "..img:size(3)
  local hmarg=img:size(2)-h
  local wmarg=img:size(3)-w
  assert(hmarg>=0 and wmarg>=0,"requested oversize crop. "..errmsg)
  local hoff=math.floor(math.random()*hmarg)
  local woff=math.floor(math.random()*wmarg)
  img=image.crop(img, woff,hoff, woff+w,hoff+h)
  if lbl then lbl=image.crop(lbl, woff,hoff, woff+w,hoff+h) end
  return img,lbl
end

function NetworkTraining.jitterImgLbl3(img,lbl,d,expand)
  local sz=torch.LongTensor(img:size())
  sz=sz:double()
  local marg=sz:narrow(1,2,sz:size(1)-1)-d
  if not expand then
    assert(torch.all(marg:ge(0)),"requested oversize crop. ")
  end
  local newsz=sz:clone()
  newsz:narrow(1,2,newsz:numel()-1):copy(d)
  local newimg=img.new(newsz:long():storage()):zero()
  local newlbl=lbl.new(d:long():storage()):zero()
  local marg2=marg:clone()
  marg2:maskedFill(marg:lt(0),0)
  local off=torch.rand(d:size()):cmul(marg2):floor()
  local srcind={}
  local dstind={}
  for i=1,d:numel() do
    local sind=math.max(1,math.floor(-marg[i]/2))
    local szi=math.min(sz[i+1],d[i])
    table.insert(dstind,{sind,sind+szi-1})
    table.insert(srcind,{off[i]+1,off[i]+szi})
  end
  newlbl[dstind]:copy(lbl[srcind]:clone())
  table.insert(dstind,1,{})
  table.insert(srcind,1,{})
  newimg[dstind]:copy(img[srcind]:clone())
  return newimg,newlbl
end

function NetworkTraining.augmentColor(img,max_z)
  --[[
    as in full-resolution residual networks, appendix A
  --]]
  local d=img:size(1)
  local maxz=max_z or 0.02
  local z_over_sqrt2=(torch.rand(d)-0.5)*maxz*math.sqrt(2)
  local gamma=torch.log(z_over_sqrt2+0.5):cdiv(torch.log(-z_over_sqrt2+0.5))
  for i=1,d do
    img[i]:pow(gamma[i])
  end
  return img
end

function NetworkTraining.augmentIntensity(img,max_z)
  --[[
    as in full-resolution residual networks, appendix A
  --]]
  local maxz=max_z or 0.02
  local z_over_sqrt2=(math.random()-0.5)*maxz*math.sqrt(2)
  local gamma=math.log(z_over_sqrt2+0.5)/math.log(-z_over_sqrt2+0.5)
  img:pow(gamma)
  return img
end


function NetworkTraining.myCrop(img,inds)
  local outsize={}
  local dstinds={}
  local srcinds={}
  for k,v in pairs(inds) do
    if #v ==0 then
      table.insert(dstinds,{})
      table.insert(srcinds,{})
      outsize[k]=img:size(k)
    else
      assert(#v==2) 
      assert(v[2]>=v[1])
      local srcbeg=math.max(1,v[1])
      local srcend=math.min(img:size(k),v[2])
      local dstbeg=math.max(1,-v[1]+2)
      local dstend=dstbeg+srcend-srcbeg
      table.insert(dstinds,{dstbeg,dstend})
      table.insert(srcinds,{srcbeg,srcend})
      outsize[k]=v[2]-v[1]+1
    end
  end
  local crop=img.new(torch.Tensor(outsize):long():storage()):zero()
  crop[dstinds]:copy(img[srcinds])
  return crop
end
