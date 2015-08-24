
function retrieveYUVImage(n, train)
  require 'image'
  require 'torch'
  dataLocT7 ='../../MSc_Data/NYU_V2/Torch/320_240/'


  
  

  
  
  width = 320
  height = 240
  nChannels = 3
  
  trInfo = torch.load('dataloc..Train_Info.t7');
  randIndeces = trInfo.randIndeces
  trIndeces = trInfo.trIndeces
  teIndeces = trInfo.teIndeces
  
  if train then
    imagesFileName = 'NYU_V2_Train_nYUV.t7'
    nLabel = trIndeces[n]
  else
    imagesFileName = 'NYU_V2_Train_nYUV.t7'
    nLabel = teIndeces[n]
  end
  local labelsFileName = 'NYU_V2_L5.t7'

  local imagesStorage = torch.FloatStorage(dataLocT7..imagesFileName,true)
  local labelsStorage = torch.ByteStorage(dataLocT7..labelsFileName, true)
  
  
  offset = 1 + (n-1)*nChannels*height*width
  local im = torch.FloatTensor(imagesStorage, offset, torch.LongStorage{nChannels,height,width})
  im2 = im:clone()
  
  
  offsetLabel = 1 + (nLabel-1)*height*width
  local lab = torch.ByteTensor(labelsStorage, offsetLabel, torch.LongStorage{height,width})
  labels = lab:clone()
  collectgarbage()
  
  return im2,labels
  
    --channels = {'y','u','v'}
    --for j,channel in ipairs(channels) do
      --im2[{{j},{},{}}]:mul(sigma[j])
      --im2[{{j},{},{}}]:add(mu[j])
    --end
    --im2 = image.yuv2rgb(im2)
    --image.save('im.jpeg',im2:div(255))
  end
  