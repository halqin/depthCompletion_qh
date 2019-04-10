% this pretrain script is used for evaluate test set and model  
% the input set is testing set  

[input_name,model_name] = pretrain2_path();
load(input_name);
load(model_name);
gpu_enable = 0;
num_im = 100; %the number of test images  
net.forward(95).args{1,4} = 'mae';

net = Net(net);
if strcmpi('PCWIN64',computer('arch')) 
    gpu_enable = 1;
    net.move('gpu');
end    
error = 0;

data(:,:,1:3,:) = single(imdb.images.data(:,:,1:3,:))/255;% normalize batch to [0,1]
data(:,:,4,:) = single(imdb.images.data(:,:,4,:))/80; 
% labels = imdb.images.labels(:,:,:,:);

for i = 1:num_im
     if gpu_enable
        net.eval({'images', gpuArray(data(:,:,:,i)), 'labels', gpuArray( imdb.images.labels(:,:,:,i))},'test');
     else 
         net.eval({'images', data(:,:,:,i), 'labels', imdb.images.labels(:,:,:,i)},'test');
     end
     
     error = error + gather( net.getValue('loss1'));
end 

ave_error = error/num_im;