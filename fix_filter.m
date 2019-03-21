clear all;
close all;

load('D:\convnet\model_result\models\demo\net-epoch-200.mat');
net = Net(net);
load('D:\convnet\depthCompletionNet-master\data\imdb_sparse_500interpo.mat');
imdb.images.data(:,:,4,:) = single(imdb.images.data(:,:,4,:)/80);

net.eval({'images',imdb.images.data(:,:,:,1), 'labels', single(imdb.images.labels(:,:,1,1))},'forward');
 
resultVia = net.getValue('prediction');
imagesc(resultVia)