clear all;
load('/Users/Hall/convnn/depthCompletionNet/imdb_sparse_500morph_test.mat');
val_im = 100; %select 100 images as validation set 

size_= size(imdb.images.data);

% val_vac = randi([1, size_(4)], val_im, 1);
val_vac = randperm(500,100);

imdb_new.images.data = zeros([288 1280 4 val_im], 'single');
imdb_new.images.labels = zeros([288 1280 1 val_im], 'single');

for i = 1:val_im
%     aa =[aa elm] ;
    imdb_new.images.data(:,:,:,i) = imdb.images.data(:,:,:,val_vac(i));
    imdb_new.images.labels(:,:,:,i) = imdb.images.labels(:,:,:,val_vac(i));
end 
clear imdb;
imdb.images.data = imdb_new.images.data;
imdb.images.labels = imdb_new.images.labels;
clear imdb_new;

save('/Users/Hall/convnn/depthCompletionNet/morph_test100.mat', 'imdb'); 
