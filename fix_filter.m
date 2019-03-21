% clear all;
% load('/Users/Hall/convnn/depthCompletionNet/imdb_sparse_500interpo.mat');
% 
% % images = imdb.images.data;
% % labels = imdb.images.labels; 
% 
% 
%     for i = 1:100
%        imdb.images.data(:,:,4,i) = 80*imbilatfilt(imdb.images.data(:,:,4,i)/80);
% %         images(:,:,4,i) = imdiffusefilt(images(:,:,4,i));
% %         images(:,:,4,i) = imguidedfilter(images(:,:,4,i));
%         
%     end
%     
% save('./imdb_sparse_bilat.mat','imdb','-v7.3');
% 
% 
% for i in 10:
%     
% end 


clear all;
close all;

load('/Users/Hall/convnn/depthCompletionNet/models/net-epoch-200-2003.mat') ;
net=Net(net);
load('/Users/Hall/convnn/depthCompletionNet/imdb_sparse_100interpo.mat');
imdb.images.data(:,:,4,:) = single(imdb.images.data(:,:,4,:)/80);
imdb.images.data(:,:,1:3,:) = single(imdb.images.data(:,:,1:3,:)/255);

N =3 ;
error = 0; 
for i =1: N
%     imdb.images.data(:,:,4,i) = imdiffusefilt(imdb.images.data(:,:,4,i));
    net.eval({'images', imdb.images.data(:,:,:,i), 'labels', single(imdb.images.labels(:,:,1,i))},'test');
    error = error + net.getValue('loss1'); 
end

ava_error = error/N;

