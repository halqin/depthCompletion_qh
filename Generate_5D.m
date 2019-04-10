% the script is used for generating 5D image data
clear all;
cc= load('f:\convnet\data\morph\imdb_sparse_100morph_test.mat');
% load();
bb = load('f:\convnet\data\morph_anis\imdb_sparse_100ansi_test2.mat');

new_size = 100;
imdb.images.data = zeros([288 1280 5 new_size],'single');
imdb.images.labels = zeros([288 1280 1 new_size],'single');
% imdb.images.set = zeros([new_size, 1],'single');

 % imdb_new.images.labels = zeros([288 1280 1 new_size],'single');
imdb.images.data = cat(3, cc.imdb.images.data(:,:,1:4,:), bb.imdb.images.data(:,:,4,:));
clear bb;
imdb.images.labels = cc.imdb.images.labels;
% imdb.images.set = cc.imdb.images.set;

clear aa;
save('f:\convnet\data\test100_5D.mat','imdb','-v7.3');

