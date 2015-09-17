-- Labeling model using the superpixels approach.

require 'imgraph'
require 'nn'

print('Defining Labelling Neural Network')

nscales = 3
nHidden = 1024
nInputs = 256 * nscales
nOutputs = 5

labelNN = nn.Sequential()

labelNN:add(nn.JoinTable(2,2))
labelNN:add(nn.Linear(nInputs,nHidden))
labelNN:add(nn.Tanh())
labelNN:add(nn.Linear(nHidden,nOutputs))
labelNN:add(nn.LogSoftMax())

print(labelNN)

criterion = nn.CrossEntropyCriterion()