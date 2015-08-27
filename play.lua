require 'nn'
--require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

-- Define parameters
nfeats = 3
width = 320
height = 240
nPixels = width*height

nstates = {16,64,256}
filtsize = 7
poolsize = 2
pad = (filtsize-1)/2
nClasses = 5

------------------------------------------------------------------------------
-- Define connectivity maps for SpatialConvolutionMap

ConMatrix1 = torch.Tensor(2,22) -- Connect three input Feature Maps to 16 Filters
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

------------------------------------------------------------------------------
-- Construct the model
convModel = nn.Sequential()

-- stage 1: filter bank, tanh, max pooling
--convModel:add(nn.SpatialConvolutionMM(nfeats,nstates[1], filtsize, filtsize))
convModel:add(nn.SpatialConvolutionMap(ConMatrix1,7,7))
convModel:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
convModel:add(nn.Tanh())
convModel:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 2: filter bank, tanh, max pooling
convModel:add(nn.SpatialConvolutionMap(ConMatrix2, filtsize, filtsize))
convModel:add(nn.SpatialZeroPadding(pad,pad,pad,pad))
convModel:add(nn.Tanh())
convModel:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 3 : filter bank
convModel:add(nn.SpatialConvolutionMap(ConMatrix3, filtsize, filtsize))
convModel:add(nn.SpatialZeroPadding(pad,pad,pad,pad))

-- Upsample to original size
--convModel:add(nn.SpatialUpSamplingNearest(4))

------------------------------------------------------------------------------
--Mock Training

convModel:training()
parameters,gradParameters = convModel:getParameters()
gradParameters:zero()

input = torch.randn(nfeats,height,width)
output = convModel:forward(input)

------------------------------------------------------------------------------
-- Test for NaN
if(output:ne(output):sum()>0) then
  
  print('NANs in output: '..output:ne(output):sum())
else 
  print('No NaNs')

end