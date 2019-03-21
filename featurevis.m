clear all;
load('/Users/Hall/convnn/depthCompletionNet/models/net-epoch-1000.mat');

figure;
resultViz = net.vars{net.getVarIndex('prediction')};

imagesc(resultViz);
title('Output');