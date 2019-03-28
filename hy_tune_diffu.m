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
M = 3; % the types  of filters 
ave_error = 0;
error = 0;
hy_len = 100; % the length of the hyperparameter vector
val_im = 3; % the number images used for validation 

% sigma_vec = 1+(10-1)* rand(hy_len,1); %DegreeOfSmoothing
% % r = a + (b-a).*rand(N,1) -----> (a,b) 
% smooth_vec = 1+(70-1)* rand(hy_len,1); % spatialSigma

grandTh_vec = 5+(15-5)* rand(hy_len,1); %GradientThreshold

numIt_vec = randi([1,15], hy_len, 1); %NumberOfIterations
con_vec = ["maximal", "minimal"]; %Connectivity

size_ = size(imdb.images.data);
val_vec = randi([1,size_(4)],val_im,1); %genreate 100 random image from images set for validation 

for j = 1:hy_len
    grandTh = grandTh_vec(j);
    numIt = numIt_vec(j);
    con = con_vec(randi([1,2],1));
    
%     I_filt= imbilatfilt(I,'DegreeOfSmoothing', smooth, 'SpatialSigma', sigma);
%     error = I-I
    for i = 1:val_im
         image_index= val_vec(i);
         imdb_new.images.data(:,:,4,image_index) = imdiffusefilt(imdb.images.data(:,:,4,image_index),'GradientThreshold', grandTh, 'NumberOfIterations', numIt, 'Connectivity',con);         
         net.eval({'images', imdb_new.images.data(:,:,:,image_index), 'labels', single(imdb.images.labels(:,:,1,image_index))},'test');
         error = error + net.getValue('loss1'); 
    end 
     ave_error = [ave_error error/N];
     error = 0;
end 

figure;
x_axis = 1:(hy_len+1);
plot(x_axis, ave_error); 
 [pos, val] = max(ave_error)
 
% figure;
% subplot(2,1,1);
% imshow(I);
% subplot(2,1,2);
% imshow(I_filt);
[pos, val] = max(ave_error)
sigma_vec(val)
smooth_vec(val)

