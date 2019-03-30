clear all;
close all;
[input_name, model_name] = pretrain_path();
load(input_name) ;
load(model_name);
net=Net(net);
% inputs = {'images',gpuArray(single(images(:,:,1:4,:))),'labels',gpuArray(single(labels))} ;

N =30; % the image index want to show  
data = imdb.images.data(:,:,:,N);
labels = imdb.images.labels(:,:,1,N);

data(:,:,1:3,:) = single(data(:,:,1:3,:))/255;% normalize batch to [0,1]
data(:,:,4,:) = single(data(:,:,4,:))/80; 

 net.move('gpu');   
 net.eval({'images', gpuArray(data), 'labels', gpuArray(labels)},'forward');
%  
% net.eval({'images', data, 'labels', labels},'forward');

figure(4);
subplot(2,1,1)
resultViz = gather( net.getValue('prediction'));

imagesc(resultViz);
title('Output');
subplot(2,1,1);
imagesc(imdb.images.data(:,:,4,N)/80);
title('Input');

% newValue = gather(sum(net.vars{net.forward(sel(1)).outputVar(1)}(:))) ; % Qh_the newValue is new inferenceScores       
loss1_value = gather(sum(net.getValue('loss1')));

X = ['The loss1 value is ', num2str(loss1_value)];
disp(X);