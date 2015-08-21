----------------------------------------------------------------------
-- This script aims at shortening the loading time of images to be fed to the network by the use of memory mapping
-- Three different files
-- Images: bytes
-- depths: floats
-- Labels: shorts

--Osama Soltan
--13.08.2015
----------------------------------------------------------------------
require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

local matio = require 'matio'	-- For the conversion of Matlab files to torch 
		  		-- (the NYU-V2 dataset is available as a .mat file)
----------------------------------------------------------------------

dataLocMat = '../../../../MSc_Data/NYU_V2/MATLAB/' -- Specify the relative location of the dataset
dataLocT7 = '../../../../MSc_Data/NYU_V2/Torch/'
-- filename = 'nyu_depth_v2_labeled.mat'
nPartitions = 20

channels = {'y','u','v'}

-- Info about the dataset
nImages = 1449
width = 640
height = 480
nChannels = 3

imagesFileName = 'NYU_V2_RGB.t7'
depthsFileName = 'NYU_V2_D.t7'
labelsFileName = 'NYU_V2_L5.t7'

imagesFileSize = nImages*width*height*nChannels
depthsFileSize = nImages*width*height
labelsFileSize = nImages*width*height

----------------------------------------------------------------------

--Allocate space for the images (107 Torch overhead)
print('Allocating files on disk')
--torch.save(dataLocT7..imagesFileName, torch.ByteTensor(imagesFileSize - 107))
--torch.save(dataLocT7..depthsFileName, torch.FloatTensor(depthsFileSize - 107))
--torch.save(dataLocT7..labelsFileName, torch.ByteTensor(labelsFileSize - 107))
--os.execute('ls -l '..imagesFileName)
imagesStorage = torch.ByteStorage(dataLocT7..imagesFileName, true,imagesFileSize)
depthsStorage = torch.FloatStorage(dataLocT7..depthsFileName, true,depthsFileSize)
labelsStorage = torch.ByteStorage(dataLocT7..labelsFileName, true,labelsFileSize)

----------------------------------------------------------------------
imagesOffset = 1
depthsOffset = 1
labelsOffset = 1

for i = 1, nPartitions do
  print('loading part '..i)
  if i > 10 then
    x = i-10
    matfilename = 'Training_Dataset_Part'..x..'.mat'
    
  else
    matfilename = 'Testing_Dataset_Part'..i..'.mat'
  end
  
  
	dataset = matio.load(dataLocMat..matfilename)
  
  print('Transposing')
  -- Transpose dataset to torch format: N*K*H*W
  dataset.images = dataset.images:transpose(1,4)
  dataset.images = dataset.images:transpose(2,3)
  images = dataset.images:transpose(3,4):clone();
  
  dataset.depths = dataset.depths:transpose(1,3)
  depths = dataset.depths:transpose(2,3):clone()
  
  dataset.labels = dataset.labels:transpose(1,3)
  labels = dataset.labels:transpose(2,3):clone()
  
  dataset = nil;
  collectgarbage()
  
  NBatch = #images
	NBatch = NBatch[1]
  
  print('getting pointers to sections')
  -- Get pointers to the relevant section of the files on disk.
  imagesChunkStorage = torch.ByteStorage(imagesStorage,imagesOffset,NBatch*nChannels*width*height) 
  depthsChunkStorage = torch.FloatStorage(depthsStorage,depthsOffset,NBatch*width*height)
  labelsChunkStorage = torch.ByteStorage(labelsStorage,labelsOffset,NBatch*width*height)
  
  print('Transferring')
  -- Transfer the dataset to the t7 files.
  imagesChunkStorage:copy(images:storage())
  depthsChunkStorage:copy(depths:storage())
  labelsChunkStorage:copy(labels:storage())
  
  print('updating offsets')
  --update offsets
  imagesOffset = imagesOffset + (NBatch * nChannels * width * height)
  depthsOffset = depthsOffset + (NBatch * width * height)
  labelsOffset = labelsOffset + (NBatch * width * height)
  
  
  end