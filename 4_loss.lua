----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- loss functions:
--   + negative-log likelihood, using log-normalized output units (SoftMax)
--   + mean-square error
--   + margin loss (SVM-like)
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

----------------------------------------------------------------------
-- 14-class problem

--noutputs = 5
--width = 160
--height = 120
--nPixels = width*height

--criterion = nn.ParallelCriterion()
--for i = 1,nPixels do
  
  -- The loss works like the MultiMarginCriterion: it takes
   -- a vector of classes, and the index of the grountruth class
   -- as arguments.
  --criterion:add(nn.CrossEntropyCriterion())
  criterion = nn.CrossEntropyCriterion()
--end
