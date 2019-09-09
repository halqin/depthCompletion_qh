%plot the learning curve 
close all;
clearvars;
% KNN = load('/Users/Hall/convnn/depthCompletionNet/models/KNN/net-epoch-200-KNN.mat');
% morp = load('/Users/Hall/convnn/depthCompletionNet/models/morp/net-epoch-200-morp.mat');
% natural = load('/Users/Hall/convnn/depthCompletionNet/models/Natual/net-epoch-200-natual.mat');
% linear = load('/Users/Hall/convnn/depthCompletionNet/models/linear/net-epoch-200-linear.mat');
% KNN = load('D:\convnet\model_result\models\demoKNN\net-epoch-200.mat');
% morp = load('D:\convnet\model_result\models\demo_morp\net-epoch-200.mat');
% aniso = load('D:\convnet\model_result\models\morp_anitsot\net-epoch-200.mat');
[a, b] = learning_curve_path();

a1 = [a.stats.train.loss1];
b1 = [b.stats.train.loss1];
%c1= [c.stats.train.loss1];
%d1 = [d.stats.train.loss1];
%e1 = [e.stats.train.loss1];



figure(2);
%plot(a1,'r+');
hold on
%plot(b1,'co');
% plot(natural1,'b*');
% plot(linear1,'gx' );
%plot(c1, 'b*');

% smooth ______________________________
x = (1:1:180);
aa1 = smooth(x, a1,0.3,'rloess');
plot(x,aa1, 'r','LineWidth',2);

bb1 = smooth(x, b1,0.3,'rloess');
plot(x,bb1, 'c','LineWidth',2);

% cc1 = smooth(x, c1,0.3,'rloess');
% plot(x,cc1,'b', 'LineWidth',2);
% 
% dd1 = smooth(x, d1,0.3,'rloess');
% plot(x,cc1,'b', 'LineWidth',2);
% ______________________________________
% Normal 
ylim([0,4]);
%plot(a1);
%plot(b1);
% plot(c1);
% plot(d1);


%----------------------------------------------------
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

lgd = legend('With mask', 'Without mask');
lgd.FontSize = 12;
% legend('Sparse', 'Morph', 'Aniso+Mask');
ts = title('Learning curve for different setting schedule');
ts.FontSize = 12; 
xlabel('Epochs');
ylabel('Loss');
grid on;
set(0,'DefaultLineLineWidth',2);


% 
% plot(KNN1,'r');
% hold on
% plot(morp1,'c');
% % plot(natural1,'b*');
% % plot(linear1,'gx' );
% plot(aniso1, 'b');
