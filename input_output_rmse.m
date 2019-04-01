% calculate the average error of interpolation vs GD
% calculate the average eror of CNN output vs GD
<<<<<<< HEAD
% clear all;
[input_name, model_name, gpuSet, save_name]= input_output_path();
=======
clear all;
[input_name, model_name, save_name, gpuSet]= input_output_path();
>>>>>>> add18fc9f8cc57ce926247e85a4baa361ae195b2

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
        
%          error_in = input_error(imdb.images.data(:,:,4,i), imdb.images.labels(:,:,1,i));
%          error_inList = [error_inList error_in];
         
         error_cnn = output_error(imdb.images.data(:,:,:,i), imdb.images.labels(:,:,1,i), net);
         error_cnnList = [error_cnnList error_cnn]; 
end 

% rmse_plot(error_inList, error_cnnList);
% save(save_name,'error_cnnList','error_inList');



function error_in = input_error(input_data, input_labels )
%          input_data = imdb.images.data(:,:,4,i) ;
%          input_labels =  imdb.images.labels(:,:,1,i));
         instanceWeights = input_labels ~= 0 ;
         
         t = (80* input_data- input_labels) .^ 2 ;
         y = instanceWeights .*t;
         y = sum(y);
         error_in = y/sum(sum(instanceWeights));
         error_in = sqrt(sum(error_in));
<<<<<<< HEAD
         
end

function error_cnn = output_error(input_data, input_labels, net)
         net.eval({'images',gpuArray( input_data), 'labels', gpuArray(single(input_labels))},'test');
%          cnn_out = gather(net.getValue('prediction'));
=======
         error_inList = [error_inList error_in];
         if gpuSet
            net.eval({'images',gpuArray( imdb.images.data(:,:,:,i)), 'labels', gpuArray(single(imdb.images.labels(:,:,1,i)))},'test');
         else
            net.eval({'images', imdb.images.data(:,:,:,i), 'labels', single(imdb.images.labels(:,:,1,i))},'test');
         end
         cnn_out = gather(net.getValue('prediction'));
         error_cnn = gather(net.getValue('loss1')); 
end



function rmse_plot(error_inList, error_cnnList)
    plot(error_inList);
    hold on
    plot(error_cnnList);
    hold off
end 
% save(save_name,'error_cnnList','error_inList');
% x = 1:1:100;
% p1 = plot(error_inList(2:101), "r+");
% hold on
% p2 = plot(error_cnnList(2:101), "b*");
% pp1 = smooth(x,error_inList(2:101), 0.1, 'rloess' );
% plot(x, pp1, 'r', 'LineWidth', 2);
% pp2 = smooth(x,error_cnnList(2:101), 0.1, 'rloess' );
% plot(x, pp2, 'b', 'LineWidth', 2);
% hold off
% plot(error_inList(2:101), 'r');
% hold on
% plot(error_inList(2:101), 'r*');
% plot(error_cnnList(2:101), 'bo');
% plot(error_cnnList(2:101),'b');

