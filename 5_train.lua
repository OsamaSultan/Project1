----------------------------------------------------------------------
-- This script demonstrates how to define a training procedure,
-- irrespective of the model/loss functions chosen.
--
-- It shows how to:
--   + construct mini-batches on the fly
--   + define a closure to estimate (a noisy) loss
--     function, as well as its derivatives wrt the parameters of the
--     model to be trained
--   + optimize the function, according to several optmization
--     methods: SGD, L-BFGS.
--
-- Clement Farabet
----------------------------------------------------------------------
require('mobdebug').start()
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'
require 'image' -- an optimization package, for online and batch methods
dofile 'retrieveYUVImage.lua'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
   cmd:option('-learningRate', 10e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- CUDA?
--if opt.type == 'cuda' then
  -- model:cuda()
   --criterion:cuda()
--end
--dataLocT7 = '../../Data/Torch/'
--dataLocT7 ='../../MSc/MSc_Data/NYU_V2/Torch/320_240/'
--trInfo = torch.load('dataloc..Train_Info.t7');




--trIndeces = trInfo.trIndeces
trsize = 795
--width = 640
--height = 480

--nChannels = 3

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
classes = {'1','2','3','4','5'}
nClasses = 5
--for i = 1,nClasses do
  --table.insert(classes,tostring(i))
--end


-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
--if model then
    --convModel = model:get(1)
    --linModel = model:get(2)
   parameters,gradParameters = convModel:getParameters()
   parametersLin,gradParametersLin = linModel:getParameters()
--end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end

----------------------------------------------------------------------
print '==> defining training procedure'

function train()
 

--local imagesFileName = 'NYU_V2_Test_nYUV.t7'
--local imagesStorage = torch.FloatStorage(dataLocT7..imagesFileName,true)
--local labelsFileName = 'NYU_V2_L5.t7'
--local labelsStorage = torch.ByteStorage(dataLocT7..labelsFileName, true)

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   convModel:training()
   linModel:training()
   -- shuffle at each epoch
   shuffle = torch.randperm(795)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trsize,opt.batchSize do
      -- disp progress
      
      xlua.progress(t, trsize)

      -- create mini batch
      local inputs = {}
      local targets = {}
      collectgarbage()
      for i = t,math.min(t+opt.batchSize-1,trsize) do
         -- load new sample
         
         n = shuffle[i]
         --offset = 1 + (n-1)*nChannels*height*width
        -- im = torch.FloatTensor(imagesStorage, offset, torch.LongStorage{nChannels,height,width})
         
         imageSample,labels = retrieveYUVImage(n,1);
         --imageSample = im:clone()
         --imageSample = image.scale(imageSample,width*scale,height*scale)
         
         --labels = image.scale(labels, width*scale,height*scale,'simple')
         labels = labels:reshape(height*width*scale*scale)
         --labels = labels:double()
         
         collectgarbage()
         local input = imageSample
         local target = labels
         --if opt.type == 'double' then 
         --input = input:double()
         --elseif opt.type == 'cuda' then input = input:cuda() 
         --end
         
         table.insert(inputs, input)
         table.insert(targets, target)
         collectgarbage()
         
        
      end

      
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
        
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       --local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                         collectgarbage()
                          local output = convModel:forward(inputs[i])
                          
                          
                          -- do an epoch on the linear model using the output of the conv model as batch of inputs
                          --local err = criterion:forward(output, targets[i])
                          
                          err = trainLin(output,targets[i])
                          f = err
                          
                          
                          collectgarbage()
                          -- estimate df/dW
                          --local df_do = criterion:backward(output, targets[i])
                          convModel:backward(inputs[i], err)

                          
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                       
                       
      end

      -- optimize on current mini-batch\
      
      if optimMethod == optim.asgd then
        
         _,_,average = optimMethod(feval, parameters, optimState)
      else
        
         optimMethod(feval, parameters, optimState)
         
      end
   
   
  
  end
  

   -- time taken
   time = sys.clock() - time
   time = time / trsize
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'convModel.net')
   local filename2 = paths.concat(opt.save, 'linModel.net')
   
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, convModel)
   torch.save(filename2, linModel)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end

function trainLin(inputsLin, targetsLin)
    
    --prepare input
    
    nInputs = inputsLin:size()[1]
    -- err is a tensor containing the gradInput vector of the linear network gathered by backward() at each pixel location 
    err = torch.Tensor(inputsLin:size())
    
    
        
    local function fevalLin(x)
      
        if x ~= parametersLin then
          parametersLin:copy(x)
        end
        
        gradParametersLin:zero()
        
        local fLin = 0
        local batchSizeLin = nInputs/4
        
      for i = 1,nInputs,batchSizeLin do
        print(i)
        -- update confusion
        collectgarbage()
          local i2 = math.min(nInputs-1,i + batchSizeLin -1) 
          local outputLin = linModel:forward(inputsLin[{{i,i2},{}}])
          
          
          local errLin = criterion:forward(outputLin,targetsLin[{{i,i2}}])
          fLin = fLin + errLin
          
          confusion:batchAdd(outputLin, targetsLin[{{i,i2}}])
          
          collectgarbage()
          local df_doLin = criterion:backward(outputLin, targetsLin[{{i,i2}}])
          err[{{i,i2}}] = linModel:backward(inputsLin[{{i,i2}}], df_doLin)
        
      end
      gradParametersLin:div(nInputs)
      fLin = fLin/nInputs
      return fLin, gradParametersLin
    end
    
    
    optimMethod(fevalLin,parametersLin,optimState)
    return err
  end
