% the script is used for generating 5D image data
clear all;
cc= load('D:\convnet\depthCompletionNet-master\data\morph\imdb_sparse_500morph.mat');
% load();
bb = load('D:\convnet\depthCompletionNet-master\data\morph_anis\imdb_sparse_500ansi2.mat');

new_size = 100;
imdb.images.data = zeros([288 1280 5 new_size],'single');
imdb.images.labels = zeros([288 1280 1 new_size],'single');
imdb.images.set = zeros([new_size, 1],'single');

 % imdb_new.images.labels = zeros([288 1280 1 new_size],'single');
imdb.images.data = cat(3, cc.imdb.images.data(:,:,1:4,:), bb.imdb.images.data(:,:,4,:));
clear bb;
imdb.images.labels = cc.imdb.images.labels;
imdb.images.set = cc.imdb.images.set;

clear aa;
save('D:\convnet\depthCompletionNet-master\data\test_5D.mat','imdb','-v7.3');

