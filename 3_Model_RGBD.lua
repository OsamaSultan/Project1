-- This script specifies the architecture of the Convolutional Neural
-- Network to be used in the scene labeling process.
----------------------------------------------------------------------

require 'torch'   -- torch
--require 'image'   -- for image transforms
require 'nn'   
--require 'optim' -- provides all sorts of trainable modules/layers

torch.setdefaulttensortype('torch.FloatTensor')
--------------------------------------------------------------------------------------------------------------------------------------------
print '==> define parameters'

-- Feature map dimensions
nfeats = 4
width = 320
height = 240
--scale = 1
nPixels = width*height




--noutputs = ninputs --The outputs are feature maps with the same dimensions of the inputs
nClasses = 5 -- number of possible classes that can be predicted.

nstates = {16,64,256}
filtsize = 7
poolsize = 2
--normkernel = image.gaussian1D(7)
--------------------------------------------------------------------------------------------------------------------------------
print '==> Connectivity Matrices'
-- Connictivity Matrices for the specification of connections between the different Convolutional layers.
ConMatrix1 = torch.Tensor(2,20) -- Connect three input channels (Feature Maps) to 16 Filters
ConMatrix1[2][{{1,12}}] = torch.range(1,12)
ConMatrix1[2][{{13,16}}] = torch.range(9,12) 
ConMatrix1[2][{{17,20}}] = torch.range(13,16) 

       
ConMatrix1[1][{{1,8}}]:fill(1)-- Luminance channel connects to first 8 Filters
ConMatrix1[1][{{9,12}}]:fill(2) -- U-Color channel connects to the middle 4 filters
ConMatrix1[1][{{13,16}}]:fill(3) -- V-Color channel connects to the middle 4 filters
ConMatrix1[1][{{17,20}}]:fill(4) -- Depth channel connects to the last 4 filters
ConMatrix1 = ConMatrix1:transpose(1,2):contiguous()


-- Random Connectivity Matrix that takes 8 random feature maps from the first stage as input for each filter of the second filter bank
ConMatrix2 = nn.tables.random(nstates[1], nstates[2], nstates[1]/2)

-- Random Connectivity Matrix that takes 32 random feature maps from the second stage as input for each filter of the third filter bank
ConMatrix3 = nn.tables.random(nstates[2], nstates[3],nstates[2]/2)
--------------------------------------------------------------------------------------------------------------------------------
print '==> Model Construction'
-- Construct the model


-- First part of the model: a typical convolution network (conv+tanh+pool)
    convModel = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> Max pooling
      pad = (filtsize-1)/2
      
      --convModel:add(nn.SpatialConvolutionMM(nfeats,nstates[1], filtsize, filtsize))
      convModel:add(nn.SpatialConvolutionMap(ConMatrix1, filtsize, filtsize))
      convModel:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
      convModel:add(nn.Tanh())
      convModel:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 2 : filter bank -> squashing -> Max pooling
      convModel:add(nn.SpatialConvolutionMap(ConMatrix2, filtsize, filtsize))
      convModel:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
      convModel:add(nn.Tanh())
      convModel:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 3 : filter bank
      convModel:add(nn.SpatialConvolutionMap(ConMatrix3, filtsize, filtsize))
      convModel:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
      
      -- Upsample to original size
      convModel:add(nn.SpatialUpSamplingNearest(4))
      
     
      
      
      
    
	-- Second Part of the Model: Linear classifier:
  
  
	-- First Adjust the shape of the output of the last conv. stage so that it is arranged in one 2-D tensor. The number of rows is the number of pixels of the feature maps. And the number of columns is the number of feature maps in the output of the last stage. This means that each row contains all the features detected at one particular pixel. 
 
  
  -- Each row of the tensor will then be considered as an input to a linear transformation layer. This layer maps the features found at one pixel location to the space of all possible object classes. LinMap:F->C
  linModel = nn.Sequential()
  
     --Reshape so that each row contains all feature information at one pixel location
      linModel:add(nn.View(nstates[3],nPixels))
      linModel:add(nn.Transpose({1,2}))
      
      linModel:add(nn.Linear(nstates[3],nClasses))
      
      linModel:add(nn.LogSoftMax())  
 
 model = nn.Sequential()
 model:add(convModel)
 model:add(linModel)
    
--print '==> here is the model:'
--print(model)


