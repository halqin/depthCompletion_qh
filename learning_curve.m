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
plot(KNN1(1:150),'r+');
hold on
plot(morp1,'co');
% plot(natural1,'b*');
% plot(linear1,'gx' );
plot(aniso1, 'b*');

x = (1:1:150);
aa1 = smooth(x, KNN1,0.3,'rloess');
plot(x,aa1, 'r','LineWidth',2);

bb1 = smooth(x, morp1,0.3,'rloess');
plot(x,bb1, 'c','LineWidth',2);

cc1 = smooth(x, aniso1,0.3,'rloess');
plot(x,cc1,'b', 'LineWidth',2);

% dd1 = smooth(x, linear1,0.3,'rloess');
% plot(x,dd1,'g', 'LineWidth',2);
% [p1,~,mu] = polyfit(x, KNN1, 5);
% f1 = polyval(p1, x,[],mu);
% plot(x,f1);
% 
% [p2,~,mu] = polyfit(x, morp1, 5);
% f2 = polyval(p2, x,[],mu);
% plot(x,f2);
% 
% [p3,~,mu] = polyfit(x, aniso1, 5);
% f3 = polyval(p3, x,[],mu);
% plot(x,f3);

% ylim([0 100]);

legend('Aniso','Morph', 'AnisoTH');
% legend('Sparse', 'Morph', 'Aniso+Mask');
title('The learning curve of 3 models');
xlabel('Epochs');
ylabel('Error');

% 
% plot(KNN1,'r');
% hold on
% plot(morp1,'c');
% % plot(natural1,'b*');
% % plot(linear1,'gx' );
% plot(aniso1, 'b');
