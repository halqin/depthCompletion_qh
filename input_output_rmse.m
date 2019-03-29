% calculate the average error of interpolation vs GD
% calculate the average eror of CNN output vs GD
clear all;
load('D:\convnet\depthCompletionNet-master\data\morph_test100.mat') %load the input
load('D:\convnet\model_result\models\demo_morp\net-epoch-200'); %load pretrained model 
net = Net(net);
size_ = size(imdb.images.data);
% val_im = size_(4);
val_im = 3;
imdb.images.data(:,:,4,:) = imdb.images.data(:,:,4,:)/80;
imdb.images.data(:,:,1:3,:) = imdb.images.data(:,:,1:3,:)/255;
error =0; 
ave_error = 0;

for i = 1:val_im
         net.eval({'images', imdb.images.data(:,:,:,i), 'labels', single(imdb.images.labels(:,:,1,i))},'test');   
         cnn_out = net.getValue('prediction');
         
         instanceWeights = imdb.images.data(:,:,4,i) ~= 0 ;
         
         t = ( 80* imdb.images.data(:,:,4,i)- cnn_out) .^ 2 ;
         y = instanceWeights .*t;
         y = sum(sum(y));
        error_ = y/288*1280;
        error_ = sqrt(error_);
        
        error = error + net.getValue('loss1'); 
end 
     ave_error = [ave_error error/val_im];
     error = 0;

 t = (80*( imdb.images.data(:,:,4,i)- cnn_out) .^ 2) ;
 y = sum(sum(t));
error_ = y/288*1280;
error_ = sqrt(error_);