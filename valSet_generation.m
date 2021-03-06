
% function valSet_generation(input_name, save_name, val_mat, mode,filter)
%input_name: input data path
%save_name : output data paht
%mode : randtest/norand
%filter : bilteral, imdiff, nofilter 
[input_name, save_name, val_mat] = valSet_generation_path();
load(input_name);
num_im = 100; %select 100 images as validation set 

% size_= size(imdb.images.data);
% val_vac = randi([1, size_(4)], val_im, 1);
% val_vac = randperm(500,100);
val_vac_strc = load(val_mat);
val_vac = val_vac_strc.val_vec;
imdb_new.images.data = zeros([288 1280 4 num_im], 'single');
imdb_new.images.labels = zeros([288 1280 1 num_im], 'single');
imdb_new.images.set = zeros([num_im, 1],'single');

for i = 1:num_im        
            imdb_new.images.data(:,:,:,i) = imdb.images.data(:,:,:,val_vac(i));
            imdb_new.images.labels(:,:,:,i) = imdb.images.labels(:,:,:,val_vac(i));
end 

clear imdb;
imdb.images.data = imdb_new.images.data;
imdb.images.labels = imdb_new.images.labels;
imdb.images.set = imdb_new.images.set;
clear imdb_new;

save(save_name, 'imdb', '-v7.3'); 

 


