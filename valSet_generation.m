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

for i = 1:val_im
%     aa =[aa elm] ;
    imdb_new.images.data(:,:,1:3,i) = imdb.images.data(:,:,1:3,i);
    imdb_new.images.data(:,:,4,i) =  imdiffusefilt(imdb.images.data(:,:,4,i), 'GradientThreshold', 10, 'NumberOfIterations', 15, 'Connectivity', 'minimal');
%     imdb_new.images.data(:,:,:,i) = imdb.images.data(:,:,:,val_vac(i));
    imdb_new.images.labels(:,:,:,i) = imdb.images.labels(:,:,:,i);
end 

% imdiffusefilt(images(:,:,4,i), 'GradientThreshold', 10, 'NumberOfIterations', 15, 'Connectivity', 'minimal');
clear imdb;
imdb.images.data = imdb_new.images.data;
imdb.images.labels = imdb_new.images.labels;
clear imdb_new;

save(save_name, 'imdb', '-v7.3'); 
