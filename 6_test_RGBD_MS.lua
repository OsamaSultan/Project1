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

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
  if not nanOK then return end
  
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
    
      input,labels = retrieveYUVDImageMS(t,0);
      
      labels = labels:reshape(height*width)
      collectgarbage()
      
      local target = labels:clone()
      labels = nil
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
