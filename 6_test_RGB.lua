----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------
require('mobdebug').start()
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

--dataLocT7 = '../../Data/Torch/'
trInfo = torch.load('dataloc..Train_Info.t7');

teIndeces = trInfo.teIndeces
tesize = 654
width = 640
height = 480
nChannels = 3

--if model then
  --  convModel = model:get(1)
    --linModel = model:get(2)
   
--end
----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
  if not nanOK then return end
   -- local vars
  --local imagesFileName = 'NYU_V2_Test_nYUV.t7'
  --local imagesStorage = torch.FloatStorage(dataLocT7..imagesFileName,true)
  --local labelsFileName = 'NYU_V2_L5.t7'
  --local labelsStorage = torch.ByteStorage(dataLocT7..labelsFileName, true)
  
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,tesize do
      -- disp progress
      xlua.progress(t, tesize)
      collectgarbage()
    
      imageSample,labels = retrieveYUVDImage(t,0);
      
      labels = labels:reshape(height*width)
      collectgarbage()
      local input = imageSample[{{1,3}}]:clone()
      local target = labels:clone()
      imageSample, labels = nil, nil
      collectgarbage()
      
      
      pred = model:forward(input)
      
      confusion:batchAdd(pred, target)

      
   end

   -- timing
   time = sys.clock() - time
   time = time / tesize
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end
