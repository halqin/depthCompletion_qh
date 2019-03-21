% path = 'D:\convnet\matconvnet-1.0-beta25\contrib\autonn\haoqin\models\demo7';
% fullFileName = fullfile(folder, 'demo7.mat');
% loss_mat = load(path,'net-epoch-500.mat');
clear all;
load('D:\convnet\matconvnet-1.0-beta25\contrib\autonn\haoqin\models\demo7\net-epoch-500.mat');

train_loss  = [stats.train.loss1].';
val_loss =[stats.val.loss1].';


figure;
plot(train_loss);
hold on;
plot(val_loss);
legend('train','val');
ylabel('Loss');
xlabel('epoch');
