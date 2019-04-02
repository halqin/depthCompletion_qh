% this pretrain script is used for evaluate test set and model  

[input_name,model_name] = pretrained2();
load(input_name);
load(model_name);

net = Net(net);
net.move('gpu');
error = 0;

data(:,:,1:3,:) = single(imdb.images.data(:,:,1:3,:))/255;% normalize batch to [0,1]
data(:,:,4,:) = single(imdb.images.data(:,:,4,:))/80; 
% labels = imdb.images.labels(:,:,:,:);

for i = 1:500
     net.eval({'images', gpuArray(data(:,:,:,i)), 'labels', gpuArray( imdb.images.labels(:,:,:,i))},'test');
     error = error + gather( net.getValue('loss1'));
end 

ave_error = error/500;