%plot the learning curve 
close all;
clear all;
KNN = load('/Users/Hall/convnn/depthCompletionNet/models/KNN/net-epoch-200-KNN.mat');
morp = load('/Users/Hall/convnn/depthCompletionNet/models/morp/net-epoch-200-morp.mat');
natural = load('/Users/Hall/convnn/depthCompletionNet/models/Natual/net-epoch-200-natual.mat');
linear = load('/Users/Hall/convnn/depthCompletionNet/models/linear/net-epoch-200-linear.mat');

KNN1 = [KNN.stats.train.loss1];
morp1 = [morp.stats.train.loss1];
natural1 = [natural.stats.train.loss1];
linear1 = [linear.stats.train.loss1];

figure;
p1 = plot(KNN1,'r+');
hold on
plot(morp1,'co');
plot(natural1,'b*');
plot(linear1,'gx' );
% plot([natural.stats.train.loss1]);
x = (1:1:200);
aa1 = smooth(x, KNN1,0.1,'rloess');
plot(x,aa1, 'r','LineWidth',2);
bb1 = smooth(x, morp1,0.1,'rloess');
plot(x,bb1, 'c','LineWidth',2);
cc1 = smooth(x, natural1,0.1,'rloess');
plot(x,cc1,'b', 'LineWidth',2);
dd1 = smooth(x, linear1,0.1,'rloess');
plot(x,dd1,'g', 'LineWidth',2);
ylim([0 5])

legend('KNN', 'Morph', 'Natural', 'Linear');
title('The learning curve of 4 interpolation methods')