
load('D:\convnet\depthCompletionNet-master\data\imdb_sparse_100.mat');
for i = 1:100
[Xq,Yq] = meshgrid([1:1280],[1:384]);
imgD = imdb.images.data(:,:,4,i);
[x,y,z] = find(imgD); x = double(x); y = double(y); z = double(z);
 F = scatteredInterpolant(x,y,z,'linear'); 
Y = scatteredInterpolant(x,y,z,'nearest'); 
Zq = F(Yq,Xq);
Zw = Y (Yq,Xq);
subplot(2,1,1);
imagesc(Zq);
subplot(2,1,2);
imagesc(Zw);
end 