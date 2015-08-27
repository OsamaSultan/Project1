require 'torch'
require 'image'

dataLocT7 ='../../MSc/MSc_Data/NYU_V2/Torch/'

-- initialize some parameters
nImages = 1449
heightIn = 480
widthIn = 640
nPixelsIn = heightIn*widthIn


nChannels = 3

heightOut = 240
widthOut = 320
nPixelsOut = heightOut*widthOut


imagesFileNameIn = 'NYU_V2_RGB.t7'
depthsFileNameIn = 'NYU_V2_D.t7'
labelsFileNameIn = 'NYU_V2_L5.t7'

imagesFileNameOut = '320_240/'..imagesFileNameIn
depthsFileNameOut = '320_240/'..depthsFileNameIn
labelsFileNameOut = '320_240/'..labelsFileNameIn

imagesFileSizeIn = nImages*nChannels*nPixelsIn
depthsFileSizeIn = nImages*nPixelsIn
labelsFileSizeIn = nImages*nPixelsIn

imagesFileSizeOut = nImages*nChannels*nPixelsOut
depthsFileSizeOut = nImages*nPixelsOut
labelsFileSizeOut = nImages*nPixelsOut

imagesStorageIn = torch.ByteStorage(dataLocT7..imagesFileNameIn,true,imagesFileSizeIn)
depthsStorageIn = torch.FloatStorage(dataLocT7..depthsFileNameIn,true,depthsFileSizeIn)
labelsStorageIn = torch.ByteStorage(dataLocT7..labelsFileNameIn,true,labelsFileSizeIn)

--allocate space for output folders
--torch.save(dataLocT7..imagesFileNameOut,torch.ByteTensor(imagesFileSizeOut))
--torch.save(dataLocT7..depthsFileNameOut,torch.FloatTensor(depthsFileSizeOut))
--torch.save(dataLocT7..labelsFileNameOut,torch.ByteTensor(labelsFileSizeOut))

--get the storage of the output folders
imagesStorageOut = torch.ByteStorage(dataLocT7..imagesFileNameOut, true, imagesFileSizeOut)
depthsStorageOut = torch.FloatStorage(dataLocT7..depthsFileNameOut, true, depthsFileSizeOut)
labelsStorageOut = torch.ByteStorage(dataLocT7..labelsFileNameOut, true, labelsFileSizeOut)



--Resize each image

for i = 1,nImages do
  
  --access the files from disk
  if (i%100 == 0) then print(i) end
  
  offsetIn = 1 + (i-1)*nPixelsIn
  im = torch.ByteTensor(imagesStorageIn,1 + (i-1)*nPixelsIn*nChannels,torch.LongStorage{nChannels,heightIn,widthIn})
  dep = torch.FloatTensor(depthsStorageIn,offsetIn,torch.LongStorage{heightIn,widthIn})
  lab = torch.ByteTensor(labelsStorageIn,offsetIn,torch.LongStorage{heightIn,widthIn})
  
  -- Clone to new memory locations to avoid conflicting storages
  im1 = im:float():clone()
  dep1 = dep:clone()
  lab1 = lab:clone()
  
  --image.save('image.jpg',im1:div(255))
  
  im2 = image.scale(im1,widthOut,heightOut,'bicubic')
  im2 = im2:byte()
  dep2 = image.scale(dep1,widthOut,heightOut,'bicubic')
  lab2 = image.scale(lab1,widthOut,heightOut,'simple')
  
  offsetOut = 1 + (i-1)*nPixelsOut
  imStorage = torch.ByteStorage(imagesStorageOut,1 + (i-1)*nPixelsOut*nChannels,nPixelsOut*nChannels)
  depStorage = torch.FloatStorage(depthsStorageOut,offsetOut,nPixelsOut)
  labStorage = torch.ByteStorage(labelsStorageOut,offsetOut,nPixelsOut)
  
  imStorage:copy(im2:storage())
  depStorage:copy(dep2:storage())
  labStorage:copy(lab2:storage())
end
--i = 1
--offsetOut = 1 + (i-1)*nPixelsOut

--im = torch.ByteTensor(imagesStorageOut,offsetOut+(i-1)*nChannels,torch.LongStorage{nChannels,heightOut,widthOut})
--dep = torch.FloatTensor(depthsStorageOut,offsetOut,torch.LongStorage{heightOut,widthOut})
--lab = torch.ByteTensor(labelsStorageOut,offsetOut,torch.LongStorage{heightOut,widthOut})

--im = im:clone():float():div(255)
--dep = dep:clone():div(torch.max(dep))

--image.save('image.jpg',im)
--image.save('depth.jpg',dep)

