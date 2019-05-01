% this is one image check, see how a good imput become bad after CNN

index_im = 58;
morp = load('f:\convnet\data\morph\imdb_sparse_100morph_test.mat');
% ani = load( 'f:\convnet\data\morph_anis\imdb_sparse_100ansi_test2.mat');
ani = load( 'f:\convnet\data\morph_anis\imdb_sparse_100ansiTH_test.mat');

morp1 =morp.imdb.images.data(:,:,1:4,index_im);
morp1_label = morp.imdb.images.labels(:,:,1,index_im);
ani1 = ani.imdb.images.data(:,:,1:4,index_im);
ani1_label = ani.imdb.images.labels(:,:,1,index_im);
figure(1); 
subplot(2,1,1), imagesc(morp1(:,:,4,:)); title('Input morph');
subplot(2,1,2), imagesc(ani1(:,:,4,:));title('Input aniso');

morp1(:,:,1:3,:) = morp1(:,:,1:3,:)/255;
morp1(:,:,4,:) = morp1(:,:,4,:)/80;
ani1(:,:,1:3,:) = ani1(:,:,1:3,:)/255;
ani1(:,:,4,:) = ani1(:,:,4,:)/80;

load('F:\convnet\model_result\models\demo_morp\net-epoch-200.mat');
net = Net(net);
if strcmpi('WIN64',computer('arch')) 
    net.move('gpu');
else 
    net.move('cpu');
end
[morploss,cnn_outmorp] = evalmodel.cnnOuterror(morp1, morp1_label,net);


load('F:\convnet\model_result\models\demo_anisoTH\net-epoch-200.mat');
net = Net(net);
if strcmpi('WIN64',computer('arch')) 
    net.move('gpu');
else 
    net.move('cpu');
end
[aniloss, cnn_outani] = evalmodel.cnnOuterror(ani1,ani1_label, net);

figure(2); 
subplot(2,1,1), imagesc(cnn_outmorp);title('CNNoutput morph');
subplot(2,1,2), imagesc(cnn_outani);title('CNNoutput aniso');


[morp_outloss,morpNNout_resi] = evalmodel.inputError(cnn_outmorp/80, morp1_label);
[anis_outloss,anisNNout_resi] = evalmodel.inputError(cnn_outani/80, morp1_label);

figure(3); 
subplot(2,1,1), imagesc(anisNNout_resi);title('anis resiout');
subplot(2,1,2), imagesc(morpNNout_resi);title('morph resiout');

[morp_inloss,morpIN_resi] = evalmodel.inputError(morp1(:,:,4,:), morp1_label);
[anis_inloss,anisIN_resi] = evalmodel.inputError(ani1(:,:,4,:), morp1_label);


figure(4); 
subplot(2,1,1), imagesc(anisIN_resi);title('anis resiIN');
subplot(2,1,2), imagesc(morpIN_resi);title('morph resiIN');


%% plot 3D figure 
% 
% figure;
% x = 1:1:1280;
% y = 1:1:288;
% plot3(x,y,morp_resi);
