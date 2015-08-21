im = retrieveYUVImage(1)

local function retrieveYUVImage(indeces)
  require 'image'
  require 'torch'
  dataLocT7 = '../../Data/Torch/'
  imagesFileName = 'NYU_V2_Train_nYUV.t7'
  
  
  
  width = 640
  height = 480
  nChannels = 3
  --indeces = torch.Tensor({795})
  --nImages = indeces:size()[1]
  
  --trInfo = torch.load('dataloc..Train_Info.t7');
  --randIndeces = trInfo.randIndeces
  --trIndeces = trInfo.trIndeces
  --teIndeces = trInfo.teIndeces
  --mu = trInfo.mu
  --sigma = trInfo.sigma
  
  --imagesTensor = torch.FloatTensor(nImages,nChannels,height,width)
  imagesStorage = torch.FloatStorage(dataLocT7..imagesFileName,true)
  --for i = 1,nImages  do
    n = indeces[1]
    offset = 1 + (n-1)*nChannels*height*width
    im = torch.FloatTensor(imagesStorage, offset, torch.LongStorage{nChannels,height,width})
    im2 = im:clone()
    
    return im2
    --channels = {'y','u','v'}
    --for j,channel in ipairs(channels) do
      --im2[{{j},{},{}}]:mul(sigma[j])
      --im2[{{j},{},{}}]:add(mu[j])
    --end
    --im2 = image.yuv2rgb(im2)
    --image.save('im.jpeg',im2:div(255))
  end
  