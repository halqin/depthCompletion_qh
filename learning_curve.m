%plot the learning curve 
clearvars;
% KNN = load('/Users/Hall/convnn/depthCompletionNet/models/KNN/net-epoch-200-KNN.mat');
% morp = load('/Users/Hall/convnn/depthCompletionNet/models/morp/net-epoch-200-morp.mat');
% natural = load('/Users/Hall/convnn/depthCompletionNet/models/Natual/net-epoch-200-natual.mat');
% linear = load('/Users/Hall/convnn/depthCompletionNet/models/linear/net-epoch-200-linear.mat');
% KNN = load('D:\convnet\model_result\models\demoKNN\net-epoch-200.mat');
% morp = load('D:\convnet\model_result\models\demo_morp\net-epoch-200.mat');
% aniso = load('D:\convnet\model_result\models\morp_anitsot\net-epoch-200.mat');
[KNN, morp, aniso] = learning_curve_path();

KNN1 = [KNN.stats.train.loss1];
morp1 = [morp.stats.train.loss1];
% natural1 = [natural.stats.train.loss1];
% linear1 = [linear.stats.train.loss1];
aniso1= [aniso.stats.train.loss1];

figure;
plot(KNN1,'r+');
hold on
plot(morp1,'co');
% plot(natural1,'b*');
% plot(linear1,'gx' );
plot(aniso1, 'b*');

x = (1:1:200);
aa1 = smooth(x, KNN1,0.1,'rloess');
plot(x,aa1, 'r','LineWidth',2);
bb1 = smooth(x, morp1,0.1,'rloess');
plot(x,bb1, 'c','LineWidth',2);
cc1 = smooth(x, aniso1,0.1,'rloess');
plot(x,cc1,'b', 'LineWidth',2);
% dd1 = smooth(x, linear1,0.1,'rloess');
% plot(x,dd1,'g', 'LineWidth',2);
ylim([0 5])

legend('KNN', 'Morph', 'aniso');
title('The learning curve of 4 interpolation methods')

% 
% plot(KNN1,'r');
% hold on
% plot(morp1,'c');
% % plot(natural1,'b*');
% % plot(linear1,'gx' );
% plot(aniso1, 'b');
