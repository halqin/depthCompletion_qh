load('D:\convnet\model_result\models\morp_anitsot\net-epoch-200.mat');
load('D:\convnet\depthCompletionNet-master\data\morph_anis\imdb_sparse_500ansi.mat' );
net = Net(net);
net.move('gpu');
error = 0;

data(:,:,1:3,:) = single(imdb.images.data(:,:,1:3,:))/255;% normalize batch to [0,1]
data(:,:,4,:) = single(imdb.images.data(:,:,4,:))/80; 
% labels = imdb.images.labels(:,:,:,:);

for i = 1:500
     net.eval({'images', gpuArray(data(:,:,:,i)), 'labels', gpuArray( imdb.images.labels(:,:,:,i))},'test');
     error = error + gather( net.getValue('loss1'));
end 

ave_error = error/500;