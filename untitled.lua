--assume we have an image from the dataset
im = image.lena()
im:mul(255):byte()
--image.display{image=im,legend='Original image'}
--print(im:type())
imSize = im:size()
imSize = imSize[1] * imSize[2] * imSize[3]
print('#a:', #a, 'total size: ' .. imSize .. ' Bytes = ' .. size/1024 .. ' KByte')

-- You need to clone, otherwise the storage is the same
-- (try to remove it and see what happens)
im2= im:transpose(1,3):clone()
print('\n#im2:', #im2)

-- Allocating space for <size> Bytes (107 Torch overhead)
print('Allocating ' .. imSize .. ' Bytes')
torch.save('c.t7', torch.ByteTensor(imSize - 107))
os.execute('ls -l c.t7')

-- cStorage is a shared storage that points to the allocated file in memory
cStorage = torch.ByteStorage('c.t7', true)
cStorage:copy(im2:storage()) -- Empty the image into the allocated memory

-- Retrieve the file from mapped memory
dStorage = torch.ByteStorage('c.t7', true) -- a storage that points to the file in memory
-- Retrieve a part of the file pointed at by the storage
d = torch.ByteTensor(dStorage,1,torch.LongStorage{256,512,3}) -- offset, and dimension to be retrieved
print('\n#d:', #d)
e = d:transpose(1,3)
print('#e:', #e)
image.display{image=e,legend='Half retrieved image'}

f = torch.ByteTensor(dStorage,size/2+1,torch.LongStorage{256,512,3})
print('\n#f:', #f)                                                  

g = f:transpose(1,3)                                                
print('#g:', #g)                                                    
image.display{image=g,legend='Second half retrieved image'}
