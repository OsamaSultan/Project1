require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

dofile '3_Model.lua'
dofile '4_loss.lua'

dataLocT7 = '../../../../MSc_Data/NYU_V2/Torch/'
trInfo = torch.load('dataloc..Train_Info.t7');


trIndeces = trInfo.trIndeces
width = 640
height = 480
nChannels = 3

scale = 4

local imagesFileName = 'NYU_V2_Train_nYUV.t7'
local imagesStorage = torch.FloatStorage(dataLocT7..imagesFileName,true)
local labelsFileName = 'NYU_V2_L5.t7'
local labelsStorage = torch.ByteStorage(dataLocT7..labelsFileName, true)


n = 1
offset = 1 + (n-1)*nChannels*height*width
im = torch.FloatTensor(imagesStorage, offset, torch.LongStorage{nChannels,height,width})

imageSample1 = im:clone()
imageSample = imageSample1[{{},{1,height/scale},{1,width/scale}}]:clone()

print('Check!')
collectgarbage()
output = model:forward(imageSample)


nLabel = trIndeces[n]
offsetLabel = 1 + (nLabel-1)*height*width
lab = torch.ByteTensor(labelsStorage, offsetLabel, torch.LongStorage{height,width})         
labels = lab:clone()
labels = image.scale(labels,width/scale,height/scale,'simple');
--labels = labels[{{1,height/scale},{1,width/scale}}]:double():clone()
labels = labels:reshape(height*width/(scale^2))
--labels = labels:split(1,1)
collectgarbage()
err = criterion(output,labels)
print(err)


