require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'image'

trInfo = torch.load('dataloc..Train_Info.t7');

teIndeces = trInfo.teIndeces
tesize = 654

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function testLabel()
  
   local time = sys.clock()

 
   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   labelNN:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,tesize do
      -- disp progress
      xlua.progress(t, tesize)
      collectgarbage()
    
      im,labels = retrieveYUVDImageMS(t,0);
      labels = image.scale(labels,height/4,width/4,'simple')
      labels = labels:reshape(height*width/16):contiguous()
      collectgarbage()
      
      local target = labels:clone()
      labels = nil
      collectgarbage()
      
      
      input = convModel:forward(im)
      pred = labelNN:forward(input)
      
      confusion:batchAdd(pred, target)

      
   end

   -- timing
   time = sys.clock() - time
   time = time / tesize
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- next iteration:
   confusion:zero()
end