clear all;

load('D:\convnet\depthCompletionNet-master\data\imdb_sparse_500morph_test.mat')
load('D:\convnet\model_result\models\demo_morp\net-epoch-200.mat');
net = Net(net);
imdb.images.data(:,:,4,:) = imdb.images.data(:,:,4,:)/80;
imdb.images.data(:,:,1:3,:) = imdb.images.data(:,:,1:3,:)/255;

imdb_new.images.data =  zeros(size(imdb.images.data),'single');
imdb_new.images.data(:,:,1:3,:) = imdb.images.data(:,:,1:3,:);

size_ = size(imdb.images.data);
N =size_(4) ; % the number of images for testing 
ave_error = 0;
error = 0;
hy_len = 100; % the length of the hyperparameter vector
val_im = 100; % the number images used for validation 

sigma_vec = 1+(10-1)* rand(hy_len,1); %spatialSigma
% r = a + (b-a).*rand(N,1) -----> (a,b) 
smooth_vec = 1+(100-1)* rand(hy_len,1); % DegreeOfSmoothing



size_ = size(imdb.images.data);
val_vec = randi([1,size_(4)],val_im,1); %genreate 100 random image from images set for validation 

for j = 1:hy_len
    sigma = sigma_vec(j,1);
    smooth = smooth_vec(j,1);

    for i = 1:val_im
         image_index= val_vec(i);
         imdb_new.images.data(:,:,4,image_index) = imbilatfilt(imdb.images.data(:,:,4,image_index),'DegreeOfSmoothing', smooth, 'SpatialSigma', sigma);
         net.eval({'images', imdb_new.images.data(:,:,:,image_index), 'labels', single(imdb.images.labels(:,:,1,image_index))},'test');
         error = error + net.getValue('loss1'); 
    end 
     ave_error = [ave_error error/val_im];
     error = 0;
end 

figure;
x_axis = 2:(hy_len+1);
plot(ave_error(2:101)); 
title('Bilateral filter random search');
xlabel('epoch');
ylabel('error');
 [val, pos] = max(ave_error)
sigma_vec(pos-1)
smooth_vec(pos-1)

[val_min, pos_min] = min(ave_error(2:101))
sigma_vec(pos_min-1)
smooth_vec(pos_min-1)



