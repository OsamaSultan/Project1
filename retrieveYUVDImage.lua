function retrieveYUVDImage(n, train)
  local image = require 'image'
  require 'torch'
  dataLocT7 ='../../MSc/MSc_Data/NYU_V2/Torch/320_240/'

  collectgarbage()
  width = 320
  height = 240
  nChannels = 4
  
  trInfo = torch.load('dataloc..Train_Info.t7');
  randIndeces = trInfo.randIndeces
  trIndeces = trInfo.trIndeces
  teIndeces = trInfo.teIndeces
  sigma = trInfo.sigma
  mu = trInfo.mu  
  
  if train then
    imagesFileName = 'NYU_V2_Train_nYUV.t7'
    nLabel = trIndeces[n]
    local imagesStorage = torch.FloatStorage(dataLocT7..imagesFileName,true)
    offset = 1 + (n-1)*nChannels*height*width
    local im = torch.FloatTensor(imagesStorage, offset, torch.LongStorage{nChannels,height,width})
    im2 = im:clone()
  else
    imagesFileName = 'NYU_V2_Test_nYUV.t7'
    nLabel = teIndeces[n]
    local imagesStorageTest = torch.FloatStorage(dataLocT7..imagesFileName,true)
    offset = 1 + (n-1)*nChannels*height*width
    local im = torch.FloatTensor(imagesStorageTest, offset, torch.LongStorage{nChannels,height,width})
    im2 = im:clone()
  end
  local labelsFileName = 'NYU_V2_L5.t7'
  collectgarbage()
  
  local labelsStorage = torch.ByteStorage(dataLocT7..labelsFileName, true)
  
  
  
  offsetLabel = 1 + (nLabel-1)*height*width
  local lab = torch.ByteTensor(labelsStorage, offsetLabel, torch.LongStorage{height,width})
  labels = lab:clone()
  image = nil
  collectgarbage()
  
   --channels = {'y','u','v'}
  --for j,channel in ipairs(channels) do
    -- im2[{{j},{},{}}]:mul(sigma[j])
     --im2[{{j},{},{}}]:add(mu[j])
  --end
    --im2 = im2[{{1},{},{}}]
    --= image.yuv2rgb(im2)
    --image.save('im.jpeg',im2:div(torch.max(im2[{{1},{},{}}])))
  return im2,labels
  end
  