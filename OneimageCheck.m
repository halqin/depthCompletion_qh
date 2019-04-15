% this is one image check, see how a good imput become bad after CNN

morp = load('f:\convnet\data\morph\imdb_sparse_500morph.mat');
ani = load( 'f:\convnet\data\morph_anis\imdb_sparse_500ansi2.mat');
morp1 =morp.imdb.images.data(:,:,1:4,39);
morp1_label = morp.imdb.images.labels(:,:,1,39);
ani1 = ani.imdb.images.data(:,:,1:4,39);
ani1_label = ani.imdb.images.labels(:,:,1,39);
figure(1); 
subplot(2,1,1), imagesc(morp1(:,:,4,:)); title('Input morph');
subplot(2,1,2), imagesc(ani1(:,:,4,:));title('Input aniso');

morp1(:,:,1:3,:) = morp1(:,:,1:3,:)/255;
morp1(:,:,4,:) = morp1(:,:,4,:)/80;
ani1(:,:,1:3,:) = ani1(:,:,1:3,:)/255;
ani1(:,:,4,:) = ani1(:,:,4,:)/80;

load('F:\convnet\model_result\models\demo_morpLessIM\net-epoch-200.mat');
net = Net(net);
if strcmpi('WIN64',computer('arch')) 
    net.move('gpu');
else 
    net.move('cpu');
end
[morploss,cnn_outmorp] = evalmodel.cnnOuterror(morp1, morp1_label,net);


load('F:\convnet\model_result\models\demo_anisoLessIM\net-epoch-200.mat');
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


[morp_outloss,morp_resi] = evalmodel.inputError(cnn_outmorp/80, morp1_label);
[~,anis_resi] = evalmodel.inputError(cnn_outani/80, morp1_label);


[morp_inloss,morp_resi_in] = evalmodel.inputError(morp1(:,:,4,:), morp1_label);
[~,anis_resi_in] = evalmodel.inputError(ani1(:,:,4,:), morp1_label);


%% plot 3D figure 

figure;
x = 1:1:1280;
y = 1:1:288;
plot3(x,y,morp_resi);
