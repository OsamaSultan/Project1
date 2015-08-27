----------------------------------------------------------------------
--This Script aims at loading and preprocessing the NYU-V2 Dataset in
--order for it to be used for training and testing the indoor scene 
--parsng system to be developed. 
--
--The script is written with the aid of the torch tutorials for supervised 
--learning developed by Clement Farabet at: 
--http://code.madbits.com/wiki/doku.php?id=tutorial_supervised
--
--Osama Soltan
--24.04.2015
----------------------------------------------------------------------
require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

----------------------------------------------------------------------
-- Load entire dataset
-- Suggested solution to memory problem: 
-- + load chunks of images
-- + preprocess the images
-- + extract the needed information
-- + save the proc[{ {},j,{},{} }]essed chunks separately on disk
-- + TODO add a loop in the training process to accomodate the separate chunks

--TODO TEST RUN!!
----------------------------------------------------------------------


loadTrInfo = false
nImages = 1449
nTrImages = 795
nTeImages = 654

batchSize = 18


width = 320
height = 240
nChannels = 3

imagesFileName = 'NYU_V2_RGB.t7'
depthsFileName = 'NYU_V2_D.t7'
labelsFileName = 'NYU_V2_L.t7'

trFileName =  'NYU_V2_Train_nYUV.t7'
teFileName =  'NYU_V2_Test_nYUV.t7'

dataLocT7 ='../../MSc/MSc_Data/NYU_V2/Torch/320_240/'

trFileSize = nTrImages*nChannels*height*width
teFileSize = nTeImages*nChannels*height*width
imagesFileSize = nImages*nChannels*height*width
----------------------------------------------------------------------

-- preprocessing data
-- + map into YUV space
-- + normalize luminance locally
-- + normalize color channels globally across entire dataset

sigma = {0,0,0} --initial std 
mu = {0,0,0} --initial mean 
NTotal = 0 --initial number of images

SSE = {0,0,0} -- Sum of Squared Errors (see resources for calculation of combined std)
GSS = {0,0,0} -- Group Sum of Squares

----------------------------------------------------------------------------
--Divide indeces 1:1449 to training and testing batches
print('Dividing Indeces')

if loadTrInfo then
  trInfo = torch.load('dataloc..Train_Info.t7');
  randIndeces = trInfo.randIndeces
  trIndeces = trInfo.trIndeces
  teIndeces = trInfo.teIndeces
  mu = trInfo.mu
  sigma = trInfo.sigma
else
  randIndeces = torch.randperm(nImages);
  trIndeces = randIndeces[{{1,nTrImages}}]
  teIndeces = randIndeces[{{-nTeImages,-1}}]
end

----------------------------------------------------------------------------
print('Allocating Space on Disk')
-- Allocate sopace for the preprocessed training and tesiting datasets
--torch.save(dataLocT7..trFileName,torch.FloatTensor(trFileSize))
--torch.save(dataLocT7..teFileName,torch.FloatTensor(teFileSize))

trStorage = torch.FloatStorage(dataLocT7..trFileName,true,trFileSize)
teStorage = torch.FloatStorage(dataLocT7..teFileName,true,teFileSize)
imagesStorage = torch.ByteStorage(dataLocT7..imagesFileName,true,imagesFileSize)
----------------------------------------------------------------------------
channels = {'y','u','v'}
saveOffset = 1;

--Convert Training data to YUV, Calculate their collective mu and std, save them separately
imageCounter = 0
i = 0
while imageCounter < nTrImages do 
  if loadTrInfo then break end
  nBatch = batchSize
  if imageCounter > (nTrImages - nBatch) then
    nBatch = nTrImages - imageCounter
  end
  i1 = imageCounter + 1
  imageCounter = imageCounter + nBatch
  i2 = imageCounter
  i = i + 1; 
 
  print('Accessing and Converting Train Data Part '..i )
  indeces = trIndeces[{{i1,i2}}] 
  nBatch = indeces:size()[1]
  imagesTensor = torch.FloatTensor(nBatch,nChannels,height,width)
  for j = 1,nBatch  do
    
    n = indeces[j]
    offset = 1 + (n-1)*nChannels*height*width
    --access the index in the original RGB file
    im = torch.ByteTensor(imagesStorage, offset, torch.LongStorage{nChannels,height,width})
    im2 = im:clone()
    im2 = im2:float()
    im = image.rgb2yuv(im2)
    imagesTensor[j] = im
  end
  
  print('Calculating Mean and STD')
	for j,channel in ipairs(channels) do
 	  -- determine mean and std for each channel in this batch:
    muBatch = imagesTensor[{ {},j,{},{} }]:mean()
    sigmaBatch = imagesTensor[{ {},j,{},{} }]:std()
    varBatch = sigmaBatch^2
		-- combine mean and standard deviation with those of previous batches:
		mu[j] = (NTotal*mu[j] + nBatch*muBatch)/(NTotal + nBatch)
		
		SSE[j] = SSE[j] + varBatch * (nBatch-1)
		GSS[j] = GSS[j] + (muBatch - mu[j])^2 * nBatch
		
		NTotal = NTotal + nBatch
		var = (SSE[j] + GSS[j])/(NTotal-1)	
		sigma[j] =  torch.sqrt(var)
	end
  
  print('Saving Part '.. i)
	-- Save The training dataset on disk
  imStorage = torch.FloatStorage(trStorage,saveOffset,nBatch*nChannels*width*height)
  saveOffset = saveOffset + nBatch*nChannels*width*height
  imStorage:copy(imagesTensor:storage())
  -- clear the memory
  imagesTensor = nil
  collectgarbage()
  
end

--save the information about the preprocessong
if not loadTrInfo then
  trInfo = {randIndeces = randIndeces,trIndeces = trIndeces, teIndeces = teIndeces, mu = mu, sigma = sigma}
  torch.save('dataloc..Train_Info.t7',trInfo)
end

print('Normalization:')
-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator 
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

teSaveOffset = 1;
imageCounter = 0
i = 0
while imageCounter < nTeImages do 
  
  nBatch = batchSize
  if imageCounter > (nTeImages - nBatch) then
    nBatch = nTeImages - imageCounter
  end
  i1 = imageCounter + 1
  imageCounter = imageCounter + nBatch
  i2 = imageCounter
    i = i + 1;

  print('Access and Convert Test Data Part '..i)
  indeces = teIndeces[{{i1,i2}}] 
  nBatch = indeces:size()[1]
  imagesTensor = torch.FloatTensor(nBatch,nChannels,height,width)
  
  for j = 1,nBatch  do
    n = indeces[j]
    offset = 1 + (n-1)*nChannels*height*width
    im = torch.ByteTensor(imagesStorage, offset, torch.LongStorage{nChannels,height,width})
    im2 = im:clone()
    im2 = im2:float()
    im = image.rgb2yuv(im2)
    imagesTensor[j] = im
  end
	
  print('Normalize Test Data')
	for j,channel in ipairs(channels) do
		-- normalize each channel globally:
   	print('==>Global norm channel '..channel)
    imagesTensor[{ {},j,{},{} }]:add(-mu[j])
		imagesTensor[{ {},j,{},{} }]:div(sigma[j])
		print('==>local normalization')
		for k = 1,nBatch do
			--normalize each image locally 
			imagesTensor[{ k,{j},{},{} }] = normalization:forward(imagesTensor[{ k,{j},{},{} }])
      collectgarbage()
   	end
    
	end
  print('Save Test Data')
	-- Save The testing dataset on disk
  imStorage = torch.FloatStorage(teStorage,teSaveOffset,nBatch*nChannels*width*height)
  teSaveOffset = teSaveOffset + nBatch*nChannels*width*height
  imStorage:copy(imagesTensor:storage())
	
	-- Free the memory	
	imagesTensor = nil
	collectgarbage()
end

-- Access Training dataset (from the previously saved file)
imageCounter = 0
i = 0
trSaveOffset = 1;
while imageCounter < nTrImages do 
  nBatch = batchSize 
  if imageCounter > (nTrImages - nBatch) then
    nBatch = nTrImages - imageCounter
  end
  imageCounter = imageCounter + nBatch
  i = i + 1;
  
	-- Access Training dataset (from the previously saved file)
  print('Access Train Data part'..i)
  imagesTensor = torch.FloatTensor(trStorage,trSaveOffset,torch.LongStorage{nBatch,nChannels,height,width})
  trSaveOffset = trSaveOffset + nBatch*nChannels*height*width

	for j,channel in ipairs(channels) do	
		--normalize each channel globlly
    print('Global norm channel '..channel)
		imagesTensor[{ {},j,{},{} }]:add(-mu[j])
		imagesTensor[{ {},j,{},{} }]:div(sigma[j])
    print('local normalization')
		for k = 1,nBatch do
			--normalize each image locally      			
			imagesTensor[{ k,{j},{},{} }] = normalization:forward(imagesTensor[{ k,{j},{},{} }])
      collectgarbage()
   	end
	end
  imagesTensor = nil
	collectgarbage()
end