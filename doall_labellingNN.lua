require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

dofile 'labellingNeuralNet.lua'
dofile 'TrainLabelling.lua'
dofile 'TestLabelling.lua'

print('Training!')
while true do
  trainLabel()
  testLabel()
end
