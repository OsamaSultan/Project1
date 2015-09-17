-- Training procedure for the neural network that labels the output of the convnet

require 'imgraph'
require 'image'
require 'nn'
require 'optim'
require 'xlua'


dofile 'retrieveYUVDImageMS.lua'

width = 320
height = 240


print('Defining Training Procedure')
-- define an optimizer

trsize = 795
optimizer = optim.sgd
optimizerState = {
      learningRate = 1e-3,
      weightDecay = 0,
      momentum = 0,
      learningRateDecay = 1e-7
      }

classes = {'1','2','3','4','5'}
nClasses = 5

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

parameters, gradParameters = labelNN:getParameters()

-- define the model of the ConvNet.
print('Loading ConvModel')

model = torch.load('results_150914/model.net')
convModel = model:get(1)
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

function trainLabel()
  
  local time = sys.clock()
  epoch = epoch or 1
  
  print('==> doing epoch on training data:')
  print("==> online epoch # " .. epoch)
  
  --rearrange the training dataset randomly
  shuffle = torch.randperm(trsize)
  
  
  for i = 1,trsize do
    xlua.progress(i,trsize)
    
    --access training images
    n = shuffle[i]
    im,lab1 = retrieveYUVDImageMS(n,1)
    lab = image.scale(lab1,height/4,width/4,'simple');
    lab = lab:reshape(height*width/16):contiguous()
  
    targets = lab:clone()
    lab1,lab = nil,nil
    
    collectgarbage()

      local function feval(x)
      
        if x ~= parameters then
          parameters:copy(x)
        end
      -- reset gradients
          gradParameters:zero()
          
          inputs = convModel:forward(im);
          outputs = labelNN:forward(inputs);
          
          confusion:batchAdd(outputs,targets)
  
          err = criterion:forward(outputs,targets)
          
          derr_do = criterion:backward(outputs,targets)
          labelNN:backward(inputs, derr_do)
          
          return err, gradParameters
  
      end
  
    optimizer(feval, parameters, optimizerState)
  
  end
  
  time = sys.clock() - time
  time = time / trsize
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)
  
   local filename = paths.concat('results_LabellingNN_YUVD', 'labelNN.net')
   
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, labelNN)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
  
end
