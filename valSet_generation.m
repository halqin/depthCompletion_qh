clear all;
[input_name, save_name, val_mat] = valSet_generation_path();
load(input_name);
val_im = 500; %select 100 images as validation set 

size_= size(imdb.images.data);

% val_vac = randi([1, size_(4)], val_im, 1);
% val_vac = randperm(500,100);
val_vac_strc = load(val_mat);
val_vac = val_vac_strc.val_vec;
imdb_new.images.data = zeros([288 1280 4 val_im], 'single');
imdb_new.images.labels = zeros([288 1280 1 val_im], 'single');
imdb_new.images.set = zeros([val_im, 1],'single');

for i = 1:val_im

    imdb_new.images.data(:,:,1:3,i) = imdb.images.data(:,:,1:3,i);
    imdb_new.images.data(:,:,4,i) = 80*( imdiffusefilt(imdb.images.data(:,:,4,i)/80, 'GradientThreshold', 14, 'NumberOfIterations', 3, 'Connectivity', 'minimal'));
%     imdb_new.images.data(:,:,:,i) = imdb.images.data(:,:,:,val_vac(i));
    imdb_new.images.labels(:,:,:,i) = imdb.images.labels(:,:,:,i);
end 
imdb_new.images.set(40:val_im) = 1;
clear imdb;
imdb.images.data = imdb_new.images.data;
imdb.images.labels = imdb_new.images.labels;
imdb.images.set = imdb_new.images.set;
clear imdb_new;

save(save_name, 'imdb', '-v7.3'); 
