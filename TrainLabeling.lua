-- Training procedure for the neural network that labels the output of the convnet

require 'imgraph'
require 'nn'
require 'optim'
dofile 'retrieveYUVImage'
require 'xlua'

width = 320
height = 240



-- define an optimizer

trSize = 795
optimizer = optim.sgd
optimizerState = {
      learningRate = 10e-3,
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
model = torch.load('results/model.net')
convModel = model:get(1)

function trainLabel()
  --access training images
  
  
  shuffle = torch.randperm(trSize)
  for i = 1,trsize do
    xlua.progress(i,trSize)
    
    n = shuffle[i]
    im1,lab1 = retrieveYUVImage(n)
    lab1 = lab1:reshape(height*width)
    im = im1:clone()
    targets = lab1:clone()
    im1,lab1 = nil,nil
    
    collectgarbage()
    
    
  end

  local function feval(x)
    
    inputs = convModel:forward(im1);
    
    outputs = labelNN:forward(inputs);
    
    confusion.batchAdd(outputs,targets);
    
    err = criterion:forward(outputs,targets)
    
    
    derr_do = criterion:backward(outputs,targets)
    labelNN:backward(inputs, derr_do);
    
  end
  
  
  
end
