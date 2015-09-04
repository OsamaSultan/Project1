-- Labeling model using the superpixels approach.

require 'imgraph'
require 'nn'

nscales = 1
nHidden = 1024
nInputs = nstates[3] * nscales
nOutputs = nClasses

labelNN = nn.Sequential()

labelNN:add(nn.Linear(nInputs,nHidden))
labelNN:add(nn.Tanh())
labelNN:add(nn.Linear(nHidden,nOutputs))
labelNN:add(nn.LogSoftMax())

criterion = nn.CrossEntropyCriterion()