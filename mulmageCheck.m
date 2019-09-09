setup_autonn;
vl_setupnn;


test_data = load('F:\convnet\data\test1.mat');
load('F:\convnet\model_result\models\demo_tri_MAE\net-epoch-50.mat');
%load('F:\convnet\model_result\models\demo_morp\net-epoch-236');
dummy = load('F:\convnet\data\test_du.mat');
net = Net(net);

if strcmpi('WIN64',computer('arch')) 
    net.move('gpu');
else 
    net.move('cpu');
end

index_im = 2;  %100
test1 =test_data.imdb.images.data(:,:,1:4,index_im);
test1(:,:,1:3,:) = test1(:,:,1:3,:)/255;
test1(:,:,4,:) = test1(:,:,4,:)/80;
dummy_dg = dummy.imdb.images.labels(:,:,1,1);


 net.eval({'images',gpuArray(single( test1)), 'labels', gpuArray(single(dummy_dg))},'test');
%  net.eval({'images',gpuArray(single( test1))},'test');
cnn_out = gather(net.getValue('output'));
%cnn_out = gather (net.getValue('prediction'));
figure(2); 
subplot(2,1,1), imagesc(test1(:,:,1:3,:)); title('RGB');
subplot(2,1,2), imagesc(cnn_out);title('Depth completion');
colormap('jet');



saveas(cnn_out, '1.png')