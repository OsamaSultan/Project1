require 'imgraph'
require 'image'
require 'optim' 
require 'nn'
require 'xlua'

dofile 'retrieveYUVDImageMS.lua'

require('mobdebug').start()
torch.setdefaulttensortype('torch.FloatTensor')

width = 320
height = 240
nClasses = 5
nImages = 1449
nTrainImages = 795;
nTestImages = 654;

print('===> Loading Model')
model = torch.load('results_150914/model.net')
convModel = model:get(1)

classes = {'Unkn.','Ground','Furn.','Props','Struct.'}
confusionTest = optim.ConfusionMatrix(classes)
confusionTrain = optim.ConfusionMatrix(classes)

print('===> Changing Upsampling layers')
-- Change the upsampling scales of the model
convModel:get(1):remove(2)
convModel:get(1):remove(2)
convModel:get(1):insert(nn.View(256,height*width/16),2)

convModel:get(2):remove(2)
convModel:get(2):remove(2)
convModel:get(2):insert(nn.SpatialUpSamplingNearest(2),2)
convModel:get(2):insert(nn.View(256,height*width/16),3)

convModel:get(3):remove(2)
convModel:get(3):remove(2)
convModel:get(3):insert(nn.SpatialUpSamplingNearest(4),2)
convModel:get(3):insert(nn.View(256,height*width/16),3)

print('===> Running Superpixel labelling on all images')
for n = 1,nImages do
  xlua.progress(n,nImages)
  
  train = n <= nTrainImages
  --get the image
  if train then
    im,lab,rgbim = retrieveYUVDImageMS(n,train);
  else
    im,lab,rgbim = retrieveYUVDImageMS(n-nTrainImages,train);
  end
  
  
  -- reshape the labels into a vector
  lab = lab:reshape(height*width):contiguous()
  
  -- Pass the image through the model and get the liklioods of each pixel
  output = model:forward(im);
  predsSmall = output:transpose(1,2)
  predsSmall = predsSmall:reshape(nClasses,height/4,width/4):contiguous()
  -- Resize each channel to original image size
  likelihoods = torch.FloatTensor(nClasses, height, width)
  for k,c in ipairs(classes) do
    likelihoods[k] = image.scale(predsSmall[k],width,height)
  end

  --Perform Superpixel segmentation on the image
  imyuv = im[1]
  graph = imgraph.graph(imyuv)
  segmentedIm = imgraph.segmentmst(graph,5,15);
  components = imgraph.extractcomponents(segmentedIm,imyuv,'mask')
  nComponents = components:size()
--print('\n===>Segmented')
  -- Predict the class of each component in the segmented image
  predictedSP = torch.ByteTensor(height,width):zero()
  for i = 1,nComponents do
    collectgarbage()
    --get information of component
    top = components.bbox_top[i]
    bottom = components.bbox_bottom[i]
    h = components.bbox_height[i]
    left = components.bbox_left[i]
    right = components.bbox_right[i]
    w = components.bbox_width[i]
    patchMask = components.mask[i]:byte()
    
    --get the class liklihoods of the pixels in the in this patch
    likelihoodsPatch = likelihoods[{{},{top,bottom},{left,right}}]
    
    -- Average Liklihood of each class for the whole segment. 
    segAvgLiklihood = torch.Tensor(nClasses)
    
    for j = 1,nClasses do
      classPatch = likelihoodsPatch[j]
      segClassLiklihood = classPatch[patchMask]
      segAvgLiklihood[j] = segClassLiklihood:mean()
    end
    
    -- Compute the class with the maximum liklihood over the segment 
    liklihood,class = torch.sort(segAvgLiklihood)
    maxLiklihoodClass = class[-1]
    
    --and enforce it as a prediction for all pixels of the segment
    segPreds = patchMask * maxLiklihoodClass
    
    predictedSP[{{top,bottom},{left,right}}]:add(segPreds)
  end
  predictedSPVector = predictedSP:reshape(height*width)
  if train then
    confusionTrain:batchAdd(predictedSPVector,lab)
  else
    confusionTest:batchAdd(predictedSPVector,lab)
  end
collectgarbage()
end
print('\nResults for '..nTrainImages .. ' Training images:')
print(confusionTrain)
print('Results for '..nTestImages .. ' Testing images:')
print(confusionTest)

--colormap1 = torch.Tensor({{1,0,1,0,0},{1,0,0,1,0},{1,0,0,0,1}})
--colormap1 = colormap1:transpose(1,2):contiguous()
--col = imgraph.colorize(pred)
--colTarget = imgraph.colorize(lab:float())


--image.save('a_im.png',rgbim:float():div(255))
--image.save('a_pred.png',col)
--image.save('a_target.png',colTarget)