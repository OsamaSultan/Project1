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

local matio = require 'matio'	-- For the conversion of Matlab files to torch 
		  		-- (the NYU-V2 dataset is available as a .mat file)
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

dataLoc = '../../../../MSc_Data/NYU_V2/' -- Specify the relative location of the dataset
-- filename = 'nyu_depth_v2_labeled.mat'
nPartitions = 10


-- The images come in the format N*K*W*H (compatible with torch)

----------------------------------------------------------------------

-- preprocessing data
-- + map into YUV space
-- + normalize luminance locally
-- + normalize color channels globally across entire dataset


-- Convert all images to YUV
sigma = {0,0,0} --initial std 
mu = {0,0,0} --initial mean 
NTotal = 0 --initial number of images

SSE = {0,0,0} -- Sum of Squared Errors (see resources for calculation of combined std)
GSS = {0,0,0} -- Grohttp://www.itorch.image/up Sum of Squares
channels = {'y','u','v'}
for i = 1, nPartitions do 
	-- Load .mat file
	trFilename = 'Training_Dataset_Part'..i..'.mat'
	matTrDataset = matio.load(dataLoc..trFilename)
	
	--convert data to float
	matTrDataset.trImages = matTrDataset.trImages:float()
	
	-- determine number of images in this batch
	NBatch = #matTrDataset.trImages
	NBatch = NBatch[1]
	
	-- perform colorspace conversion for each image
	for j = 1,NBatch do
		matTrDataset.trImages[j] = image.rgb2yuv(matTrDataset.trImages[j])
	end

	for j,channel in ipairs(channels) do
 	  	-- determine mean and std for each channel in this batch:
 	 	muBatch = matTrDataset.trImages[{ {},j,{},{} }]:mean()
   		sigmaBatch = matTrDataset.trImages[{ {},j,{},{} }]:std()
		varBatch = sigmaBatch^2
	collectgarbage()
		-- combine mean and standard deviation with those of previous batches:
		mu[j] = (NTotal*mu[j] + NBatch*muBatch)/(NTotal + NBatch)
		
		SSE[j] = SSE[j] + varBatch * (NBatch-1)
		GSS[j] = GSS[j] + (muBatch - mu[j])^2 * NBatch
		
		NTotal = NTotal + NBatch
		var = (SSE[j] + GSS[j])/(NTotal-1)	
		sigma[j] =  torch.sqrt(var)
	end
	-- Save The training dataset on disk
	trFilenameTorch = 'torch/Training_Dataset_Part_'..i..'.dat'
	torch.save(dataLoc..trFilenameTorch, matTrDataset, 'binary')
	
	-- free the memory	
	matTrDataset = nil
	collectgarbage()
end

-- Local normalization
print '==> preprocessing data: normalize all three channels locally'

-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

for i = 1, nPartitions do
	--Load Testing dataset
	testFilename = 'Testing_Dataset_Part'..i..'.mat'
	matTestDataset = matio.load(dataLoc..testFilename)
	collectgarbage()
	--convert data to float
	matTestDataset.testImages = matTestDataset.testImages:float()
	
	-- determine number of images in this batch
	NBatch = #matTestDataset.testImages
	NBatch = NBatch[1]
	
	-- perform colorspace conversion for each image
	for j = 1,NBatch do
		matTestDataset.testImages[j] = image.rgb2yuv(matTestDataset.testImages[j])
	end
	
	for j,channel in ipairs(channels) do
		-- normalize each channel globally:
   		matTestDataset.testImages[{ {},j,{},{} }]:add(-mu[j])
		matTestDataset.testImages[{ {},j,{},{} }]:div(sigma[j])
		
		for k = 1,NBatch do
			--normalize each image locally      			
			matTestDataset.testImages[{ k,{j},{},{} }] = normalization:forward(matTestDataset.testImages[{ k,{j},{},{} }])
   		end
	end
	-- Save The testing dataset on disk
	testFilenameTorch = 'torch/Testing_Dataset_Part_'..i..'.dat'
	torch.save(dataLoc..testFilenameTorch, matTestDataset, 'binary')
	-- Free the memory	
	matTestDataset = nil
	collectgarbage()
	
	-- Load Training dataset again (from the previously saved file)
	matTrDataset = torch.load(dataLoc..'torch/Training_Dataset_Part_'..i..'.dat','binary')

	for j,channel in ipairs(channels) do	
		--normalize each channel globlly
		matTrDataset.trImages[{ {},j,{},{} }]:add(-mu[j])
		matTrDataset.trImages[{ {},j,{},{} }]:div(sigma[j])
		for k = 1,NBatch do
			--normalize each image locally      			
			matTrDataset.trImages[{ k,{j},{},{} }] = normalization:forward(matTrDataset.trImages[{ k,{j},{},{} }])
   		end
	end
	-- Save The training dataset on disk
	trFilenameTorch = "torch/Training_Dataset_Part_"..i..".dat"
	torch.save(dataLoc..trFilenameTorch, matTrDataset, 'binary')	
end

--TODO verify preprocessing ?? 

--image1 = Data.imageData[{{210,213},{},{},{}}]
--itorch.image(image1)


