% calculate the average error of interpolation vs GD
% calculate the average eror of CNN output vs GD
% clear all;
[input_name, model_name, gpuSet, save_name]= input_output_path();

load(model_name);
load(input_name);
net = Net(net);
if gpuSet
     net.move('gpu');   
end 
size_ = size(imdb.images.data);
% net.getValue('loss1');
% val_im = size_(4);
val_im = 100;
imdb.images.data(:,:,4,:) = imdb.images.data(:,:,4,:)/80;
imdb.images.data(:,:,1:3,:) = imdb.images.data(:,:,1:3,:)/255;
error =0; 
error_inList = 0;
error_cnnList = 0;

for i = 1:val_im
% input with fitlers 
         input_data = imdb.images.data(:,:,4,i);
%          input_data = imbilatfilt(imdb.images.data(:,:,4,i),'DegreeOfSmoothing', 3.5, 'SpatialSigma', 5.8);
%          input_data = imbilatfilt(imdb.images.data(:,:,4,i));
%          input_data = imdiffusefilt(imdb.images.data(:,:,4,i), 'GradientThreshold', 14, 'NumberOfIterations', 3, 'Connectivity','minimal');
%            input_data = imdiffusefilt(imdb.images.data(:,:,4,i));
                  
         error_in = input_error(input_data, imdb.images.labels(:,:,1,i));
         error_inList = [error_inList error_in];
         
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
         
end

function error_cnn = output_error(input_data, input_labels, net)
         net.eval({'images',gpuArray( input_data), 'labels', gpuArray(single(input_labels))},'test');
%          cnn_out = gather(net.getValue('prediction'));
         error_cnn = gather(net.getValue('loss1')); 
end



function rmse_plot(error_inList, error_cnnList)
    plot(error_inList);
    hold on
    plot(error_cnnList);
    hold off
end 


