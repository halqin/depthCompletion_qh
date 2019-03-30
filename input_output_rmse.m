% calculate the average error of interpolation vs GD
% calculate the average eror of CNN output vs GD
clear all;
[input_name, model_name, save_name, gpuSet]= input_output_path();

load(model_name);
load(input_name);
net = Net(net);
if gpuSet
     net.move('gpu');   
end 
size_ = size(imdb.images.data);
net.getValue('loss1');
% val_im = size_(4);
val_im = 100;
imdb.images.data(:,:,4,:) = imdb.images.data(:,:,4,:)/80;
imdb.images.data(:,:,1:3,:) = imdb.images.data(:,:,1:3,:)/255;
error =0; 
error_inList = 0;
error_cnnList = 0;

for i = 1:val_im
        
         instanceWeights = imdb.images.labels(:,:,1,i) ~= 0 ;
         
         t = (80* imdb.images.data(:,:,4,i)- imdb.images.labels(:,:,1,i)) .^ 2 ;
         y = instanceWeights .*t;
         y = sum(y);
         error_in = y/sum(sum(instanceWeights));
         error_in = sqrt(sum(error_in));
         error_inList = [error_inList error_in];
         if gpuSet
            net.eval({'images',gpuArray( imdb.images.data(:,:,:,i)), 'labels', gpuArray(single(imdb.images.labels(:,:,1,i)))},'test');
         else
            net.eval({'images', imdb.images.data(:,:,:,i), 'labels', single(imdb.images.labels(:,:,1,i))},'test');
         end
         cnn_out = gather(net.getValue('prediction'));
         error_cnn = gather(net.getValue('loss1')); 
         error_cnnList = [error_cnnList error_cnn]; 
end 


save(save_name,'error_cnnList','error_inList');
% x = 1:1:100;
% p1 = plot(error_inList(2:101), "r+");
% hold on
% p2 = plot(error_cnnList(2:101), "b*");
% pp1 = smooth(x,error_inList(2:101), 0.1, 'rloess' );
% plot(x, pp1, 'r', 'LineWidth', 2);
% pp2 = smooth(x,error_cnnList(2:101), 0.1, 'rloess' );
% plot(x, pp2, 'b', 'LineWidth', 2);
% hold off
plot(error_inList(2:101), 'r');
hold on
plot(error_inList(2:101), 'r*');
plot(error_cnnList(2:101), 'bo');
plot(error_cnnList(2:101),'b');
