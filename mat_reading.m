clearvars;
load('D:\convnet\depthCompletionNet-master\data\natural\imdb_sparse_1000interponatural.mat');

% size(imdb.images)
new_size = 500;

imdb_new.images.data = zeros([288 1280 4 new_size],'single');
imdb_new.images.labels = zeros([288 1280 1 new_size],'single');
imdb_new.images.set = zeros([new_size, 1],'single');


imdb_new.images.data(:,:,1:3,:) = imdb.images.data(97:384,:,1:3,501:1000); %97; 501~1000
imdb_new.images.data(:,:,4,:) = imdb.images.data(97:384,:,4,501:1000);
imdb_new.images.labels = imdb.images.labels(97:384,:,1,501:1000);
imdb_new.images.set(40:new_size) = 1; %training 
clear imdb; 
imdb.images.data = imdb_new.images.data;
imdb.images.labels = imdb_new.images.labels;
imdb.images.set = imdb_new.images.set;
clear imdb_new; 

save('D:\convnet\depthCompletionNet-master\data\natural\imdb_sparse_500natural_test.mat','imdb','-v7.3');
