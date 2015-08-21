-- This script specifies the architecture of the Convolutional Neural
-- Network to be used in the scene labeling process.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

torch.setdefaulttensortype('torch.FloatTensor')
--------------------------------------------------------------------------------------------------------------------------------------------
print '==> define parameters'

-- Feature map dimensions
nfeats = 3
width = 640
height = 480
scale = 0.5
nPixels = width*height*scale*scale
ninputs = nfeats*nPixels



noutputs = ninputs --The outputs are feature maps with the same dimensions of the inputs
nClasses = 5 -- number of possible classes that can be predicted.

nstates = {16,64,256}
filtsize = 7
poolsize = 2
normkernel = image.gaussian1D(7)
--------------------------------------------------------------------------------------------------------------------------------
print '==> Connectivity Matrices'
-- Connictivity Matrices for the specification of connections between the different Convolutional layers.
ConMatrix1 = torch.Tensor(2,22) -- Connect three input channels (Feature Maps) to 16 Filters
ConMatrix1[2][{{1,16}}] = torch.range(1,16)
ConMatrix1[2][{{17,22}}] = torch.range(11,16) 
       
ConMatrix1[1][{{1,10}}]:fill(1)-- First channel connects to first 10 Filters
ConMatrix1[1][{{11,16}}]:fill(2) -- Second channel connects to the last 6 filters
ConMatrix1[1][{{17,22}}]:fill(3) -- Third channel connects to the last 6 filters
ConMatrix1 = ConMatrix1:transpose(1,2)

-- Random Connectivity Matrix that takes 8 random feature maps from the first stage as input for each filter of the second filter bank
ConMatrix2 = nn.tables.random(nstates[1], nstates[2], nstates[1]/2)

-- Random Connectivity Matrix that takes 32 random feature maps from the second stage as input for each filter of the third filter bank
ConMatrix3 = nn.tables.random(nstates[2], nstates[3],nstates[2]/2)
--------------------------------------------------------------------------------------------------------------------------------
print '==> Model Construction'
-- Construt the model

-- a typical modern convolution network (conv+relu+pool)
      model = nn.Sequential()

      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      --model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.SpatialConvolutionMap(ConMatrix1, filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
      --model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      model:add(nn.SpatialConvolutionMap(ConMatrix2, filtsize, filtsize))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

      -- stage 3 : filter bank
      model:add(nn.SpatialConvolutionMap(ConMatrix3, filtsize, filtsize))
      pad = (filtsize-1) + math.ceil((filtsize-1)/2) + math.ceil((filtsize-1)/4)
      
      pad = math.ceil(pad/2)
      model:add(nn.SpatialZeroPadding(pad-1,pad,pad-1,pad))
      model:add(nn.SpatialUpSamplingNearest(4))
      

    -- The output of the final filter bank is a 3D tensor, with the number of generated feature maps (256) as its first dimension and the last two dimensions are the width and the height of the images. Each feature map contains the instances of one particular feature across all locations of the image.
  
	-- stage 4 : linear classifier
  
	-- First Adjust the shape of the output of the last conv. stage so that it is arranged in one 2-D tensor. The number of rows is the number of pixels of the feature maps. And the number of columns is the number of feature maps in the output of the last stage. This means that each row contains all the features detected at one particular pixel. 
 model:add(nn.Reshape(nstates[3],nPixels))
 model:add(nn.Transpose({1,2}))
  
  -- Each row of the tensor will then be considered as an input to a linear transformation layer. This layer maps the features found at one pixel location to the space of all possible object classes. LinMap:F->C
  
model:add(nn.Linear(nstates[3],nClasses))

--model:add(nn.LogSoftMax())  
 -- SoftmaxLayer = nn.Parallel(1,1) -- iterate over the rows of the input

  --for i = 1,nPixels do
    --local sq = nn.Sequential()
    --sq:add(dummyLin)
    --sq:add(nn.LogSoftMax()) -- Negative Log Likelyhood for later computation of the Cross Entropy Criterion
    --SoftmaxLayer:add(sq)
  --end
  --model:add(SoftmaxLayer)
 -- model:add(nn.Reshape(nstates[3],nPixels))
 --model:add(nn.SplitTable(1,2))

--print '==> here is the model:'
--print(model)


