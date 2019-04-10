% this  script is used for evaluate pretrain model 
function error_cnnList = evalModel(input_name, model_name, num_im, lossMethod)
% input_name: input image path
% model_name: input model path
%num_im: the number of images of test set
% lossMethod: MSE or MAE 
load(input_name);
load(model_name);
error_cnnList = zeros(1,100);
switch lossMethod
    case 'mse'
        net.forward(95).args{1,4} = 'mse'; % could choose 'MSE' or 'MAE'
    case 'mae'
        net.forward(95).args{1,4} = 'mae';
end

net = Net(net);

if strcmpi('WIN64',computer('arch')) 
    net.move('gpu');
else 
    net.move('cpu');
end    

data(:,:,1:3,:) = single(imdb.images.data(:,:,1:3,:))/255;% normalize batch to [0,1]
data(:,:,4,:) = single(imdb.images.data(:,:,4,:))/80; 
% labels = imdb.images.labels(:,:,:,:);

for i = 1:num_im
       error_cnn = evalmodel.cnnOuterror(data(:,:,:,i), imdb.images.labels(:,:,:,i), net);
       error_cnnList(i) = error_cnn; 
end 

% ave_error = sum(error_cnnList)/num_im;
% plot(error_cnnList);
end 
