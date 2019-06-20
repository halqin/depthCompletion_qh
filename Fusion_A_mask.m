function [] = Fusion_A(imdb)
% demo script for training a dummy depth completion network
% SETUP:
% run /matconvnet-1.0-beta25/matlab/vl_setupnn;
% gpuDevice(1)
% [imdb] = generate_imdb_demo([]); 
setup_autonn;
vl_setupnn;
   
%%% train %%%
try  % detect the usable of GPU 
   gpuArray(1);
   gpus=true;
catch
   gpus=[];
end 

[in, out] = U_Net_path();
% setup location for network coefficients
opts.expDir = out;
% load('D:\convnet\depthCompletionNet-master\data\morph_anis\morphani_5D.mat');
% load('F:\convnet\data\morph\imdb_sparse_500morph.mat'); 
% load('F:\convnet\data\morph_anis\imdb_sparse_500aniop.mat');
load(in);
% opts.expDir = fullfile('D:\convnet\matconvnet-1.0-beta25\contrib\autonn\haoqin\models', 'demo') ;
% load('D:\convnet\depthCompletionNet-master\depthCompletionNet-master\imdb_sparse.mat');

if gpus %select batchSize according to GPU or CPU
    gpuDevice(1);
    batchSize = 3; % gpu
else 
    batchSize = 2; % cpu
end 

opts.batchSize = batchSize; 
imdb.batchSize = opts.batchSize;
opts.gpus = gpus;


images = Input('images');

if gpus
    images.gpu = true; %mac
else 
    images.gpu = false;
end

channels = 16;
expansion = [1,2,4,4,4,8]; % the factors used to expand the channel number

% depth U-net pathway %
fsLow = [3 , 3]; padLow = floor(fsLow(1)/2);
fsMed = [3 , 3]; padMed = floor(fsMed(1)/2);
fsHigh= [3 , 3]; padHigh= floor(fsHigh(1)/2);

fsMed_simple = [7 , 7]; 
fsHigh_simple= [11 , 11]; 

R = 0.5; % dropout rate

dnMethod = 'avg'; % avg  
upMethod = 'max'; % avg        'max' | 'avg'

% morphP = 2; % 5||3
% morphSize = 3;
% leak = 0.01; % 0.01
% nMorph = 4;

entryRGB  = images(:,:,1:3,:); % the RGB channel 
entryDepth  = images(:,:,4,:); % the depth channel 


conv1 = vl_nnconv(entryDepth, 'size', [fsHigh_simple(1), fsHigh_simple(2), 1, expansion(1)*channels], 'stride',1,'pad', 5, 'hasBias', true );
conv2 = vl_nnconv(conv1, 'size', [fsMed_simple(1), fsMed_simple(2), 16, expansion(1)*channels], 'stride',1,'pad', 3, 'hasBias', true );
conv3 = vl_nnconv(conv2, 'size', [5, 5, 16, expansion(1)*channels], 'stride',1,'pad', 2, 'hasBias', true);
conv4 = vl_nnconv(conv3, 'size', [fsLow(1), fsLow(2), 16, expansion(1)*channels], 'stride',1,'pad', 1, 'hasBias', true );
conv5 = vl_nnconv(conv4, 'size', [fsLow(1), fsLow(2), 16, expansion(1)*channels], 'stride',1,'pad', 1,'hasBias', true );
Depth_branch_output = vl_nnconv(conv5, 'size', [1, 1, 16, 1], 'stride',1,'pad', 0 );

cat_in = vl_nnconcat({Depth_branch_output,entryRGB}, 3 , []);

conv1 = vl_nnconv(cat_in, 'size', [fsLow(1), fsLow(2), 4, expansion(1)*channels], 'stride',1,'pad', padLow );
relu1_1 = vl_nnrelu(conv1);
conv1_11 = vl_nnconv(relu1_1, 'size', [fsLow(1), fsLow(2), expansion(1)*channels, expansion(1)*channels], 'stride',1,'pad', padLow );
relu1_11 = vl_nnrelu(conv1_11);
conv1_111 = vl_nnconv(relu1_11, 'size', [fsLow(1), fsLow(2), expansion(1)*channels, expansion(1)*channels], 'stride',1,'pad', padLow );
relu1_111 = vl_nnrelu(conv1_111);
drop1_111 = vl_nndropout(relu1_111, 'rate', R);
pool1 = vl_nnpool(drop1_111, 2, 'method', dnMethod, 'stride', 2 , 'pad' ,0);


conv2 = vl_nnconv(pool1, 'size', [fsMed(1), fsMed(2), expansion(1)*channels, expansion(2)*channels], 'stride',1,'pad', padMed );
relu2_1 = vl_nnrelu(conv2);
conv2_11 = vl_nnconv(relu2_1, 'size', [fsMed(1), fsMed(2), expansion(2)*channels, expansion(2)*channels], 'stride',1,'pad', padMed );
relu2_11 = vl_nnrelu(conv2_11);
conv2_111 = vl_nnconv(relu2_11, 'size', [fsMed(1), fsMed(2), expansion(2)*channels, expansion(2)*channels], 'stride',1,'pad', padMed );
relu2_111 = vl_nnrelu(conv2_111);
drop2_111   = vl_nndropout(relu2_111, 'rate', R);
pool2 = vl_nnpool(drop2_111, 2, 'method', dnMethod, 'stride', 2 , 'pad' ,0);

conv3 = vl_nnconv(pool2, 'size', [fsMed(1), fsMed(2), expansion(2)*channels, expansion(3)*channels], 'stride',1,'pad', padMed );
relu3_1 = vl_nnrelu(conv3);
conv3_11 = vl_nnconv(relu3_1, 'size', [fsMed(1), fsMed(2), expansion(3)*channels, expansion(3)*channels], 'stride',1,'pad', padMed );
relu3_11 = vl_nnrelu(conv3_11);
conv3_111 = vl_nnconv(relu3_11, 'size', [fsMed(1), fsMed(2), expansion(3)*channels, expansion(3)*channels], 'stride',1,'pad', padMed );
relu3_111 = vl_nnrelu(conv3_111);
drop3_111   = vl_nndropout(relu3_111, 'rate', R);
pool3 = vl_nnpool(drop3_111, 2, 'method', dnMethod, 'stride', 2 , 'pad' ,0);

conv4 = vl_nnconv(pool3, 'size', [fsMed(1), fsMed(2), expansion(3)*channels, expansion(4)*channels], 'stride',1,'pad', padMed );
relu4_1 = vl_nnrelu(conv4);
conv4_11 = vl_nnconv(relu4_1, 'size', [fsMed(1), fsMed(2), expansion(4)*channels, expansion(4)*channels], 'stride',1,'pad', padMed );
relu4_11 = vl_nnrelu(conv4_11);
conv4_111 = vl_nnconv(relu4_11, 'size', [fsMed(1), fsMed(2), expansion(4)*channels, expansion(4)*channels], 'stride',1,'pad', padMed );
relu4_111 = vl_nnrelu(conv4_111);
drop4_111   = vl_nndropout(relu4_111, 'rate', R);
pool4 = vl_nnpool(drop4_111, 2, 'method', dnMethod, 'stride', 2 , 'pad' ,0);

conv5 = vl_nnconv(pool4, 'size', [fsHigh(1), fsHigh(2), expansion(4)*channels, expansion(5)*channels], 'stride',1,'pad', padHigh );
relu5_1 = vl_nnrelu(conv5);
conv5_11 = vl_nnconv(relu5_1, 'size', [fsHigh(1), fsHigh(2), expansion(5)*channels, expansion(5)*channels], 'stride',1,'pad', padHigh );
relu5_11 = vl_nnrelu(conv5_11);
conv5_111 = vl_nnconv(relu5_11, 'size', [fsHigh(1), fsHigh(2), expansion(5)*channels, expansion(5)*channels], 'stride',1,'pad', padHigh );
relu5_111 = vl_nnrelu(conv5_111);
drop5_1111   = vl_nndropout(relu5_111, 'rate', R);
pool5 = vl_nnpool(drop5_1111, 2, 'method', dnMethod, 'stride', 2 , 'pad' ,0);


convMix1 = vl_nnconv(pool5, 'size', [fsHigh(1), fsHigh(2), expansion(5)*channels, expansion(6)*channels], 'stride',1,'pad', padHigh );
reluMix1 = vl_nnrelu(convMix1);
convMix2 = vl_nnconv(reluMix1, 'size', [fsHigh(1), fsHigh(2), expansion(6)*channels,expansion(6)*channels], 'stride',1,'pad', padHigh );
reluMix2 = vl_nnrelu(convMix2);
dropMix = vl_nndropout(reluMix2, 'rate', R);


conv5U = vl_nnconvt(dropMix, 'size', [fsMed(1), fsMed(2), expansion(5)*channels, expansion(6)*channels], 'hasBias', true ,'Upsample', 2, 'Crop' , [padMed-1, padMed-1, padMed-1, padMed-1]);
pool5_U = vl_nnpool(conv5U, 2, 'method', upMethod, 'stride', 1 , 'pad' ,0);
relu5U_0 = vl_nnrelu(pool5_U);
cat5U = vl_nnconcat({relu5U_0,relu5_111} , 3 , []);
conv5_1U = vl_nnconv(cat5U, 'size', [fsMed(1), fsMed(2), expansion(6)*channels, expansion(5)*channels], 'stride',1,'pad', padMed );
relu5_1U = vl_nnrelu(conv5_1U);
conv5_1U1 = vl_nnconv(relu5_1U, 'size', [fsMed(1), fsMed(2), expansion(5)*channels, expansion(5)*channels], 'stride',1,'pad', padMed ); 
relu5_1U1 = vl_nnrelu(conv5_1U1);
% drop5_1U = vl_nndropout(relu5_1U1, 'rate', R);

conv4U = vl_nnconvt(relu5_1U1, 'size', [fsMed(1), fsMed(2), expansion(4)*channels, expansion(5)*channels], 'hasBias', true ,'Upsample', 2, 'Crop' , [padMed-1, padMed-1, padMed-1, padMed-1]);
pool4_U = vl_nnpool(conv4U, 2, 'method', upMethod, 'stride', 1 , 'pad' ,0);
relu4U_0 = vl_nnrelu(pool4_U);
cat4U = vl_nnconcat({relu4U_0,relu4_111} , 3 , []);
conv4_1U = vl_nnconv(cat4U, 'size', [fsMed(1), fsMed(2), expansion(5)*channels, expansion(4)*channels], 'stride',1,'pad', padMed );
relu4_1U = vl_nnrelu(conv4_1U);
conv4_1U1 = vl_nnconv(relu4_1U, 'size', [fsMed(1), fsMed(2), expansion(4)*channels, expansion(4)*channels], 'stride',1,'pad', padMed );
relu4_1U1 = vl_nnrelu(conv4_1U1);
% drop4_1U = vl_nndropout(relu4_1U1, 'rate', R);

conv3U = vl_nnconvt(relu4_1U1, 'size', [fsMed(1), fsMed(2), expansion(3)*channels, expansion(4)*channels], 'hasBias', true ,'Upsample', 2, 'Crop' , [padMed-1, padMed-1, padMed-1, padMed-1]);
pool3_U = vl_nnpool(conv3U, 2, 'method', upMethod, 'stride', 1 , 'pad' ,0);
relu3U_0 = vl_nnrelu(pool3_U);
cat3U = vl_nnconcat({relu3U_0,relu3_111} , 3 , []);
conv3_1U = vl_nnconv(cat3U, 'size', [fsMed(1), fsMed(2), expansion(4)*channels, expansion(3)*channels], 'stride',1,'pad', padMed );
relu3_1U = vl_nnrelu(conv3_1U);
conv3_1U1 = vl_nnconv(relu3_1U, 'size', [fsMed(1), fsMed(2), expansion(3)*channels, expansion(3)*channels], 'stride',1,'pad', padMed );
relu3_1U1 = vl_nnrelu(conv3_1U1);
% drop3_1U = vl_nndropout(relu3_1U1, 'rate', R);

conv2U = vl_nnconvt(relu3_1U1, 'size', [fsMed(1), fsMed(2), expansion(2)*channels, expansion(3)*channels], 'hasBias', true ,'Upsample', 2, 'Crop' , [padMed-1, padMed-1, padMed-1, padMed-1]);
pool2_U = vl_nnpool(conv2U, 2, 'method', upMethod, 'stride', 1 , 'pad' ,0);
relu2U_0 = vl_nnrelu(pool2_U);
cat2U = vl_nnconcat({relu2U_0,relu2_111} , 3 , []);
conv2_1U = vl_nnconv(cat2U, 'size', [fsMed(1), fsMed(2), expansion(3)*channels, expansion(2)*channels], 'stride',1,'pad', padMed );
relu2_1U = vl_nnrelu(conv2_1U);
conv2_1U1 = vl_nnconv(relu2_1U, 'size', [fsMed(1), fsMed(2), expansion(2)*channels, expansion(2)*channels], 'stride',1,'pad', padMed );
relu2_1U1 = vl_nnrelu(conv2_1U1);
% drop2_1U = vl_nndropout(relu2_1U1, 'rate', R);

conv1U = vl_nnconvt(relu2_1U1, 'size', [fsLow(1), fsLow(2), expansion(1)*channels, expansion(2)*channels], 'hasBias', true ,'Upsample', 2, 'Crop' , [padLow-1, padLow-1, padLow-1, padLow-1]);
pool1_U = vl_nnpool(conv1U, 2, 'method', upMethod, 'stride', 1 , 'pad' ,0);
relu1U_0 = vl_nnrelu(pool1_U);
cat1U = vl_nnconcat({relu1U_0,relu1_111} , 3 , []);
conv1_1U = vl_nnconv(cat1U, 'size', [fsLow(1), fsLow(2), expansion(2)*channels, expansion(1)*channels], 'stride',1,'pad', padLow );
relu1_1U = vl_nnrelu(conv1_1U);
conv1_1U1 = vl_nnconv(relu1_1U, 'size', [fsLow(1), fsLow(2), expansion(1)*channels, expansion(1)*channels], 'stride',1,'pad', padLow );
relu1_1U1 = vl_nnrelu(conv1_1U1);
% drop1_1U = vl_nndropout(relu1_1U1, 'rate', R);
output = 80*vl_nnconv(relu1_1U1, 'size', [1,1,expansion(1)*channels,1], 'stride',1,'pad', 0 , 'hasBias',false);
% prediction = 80*sum(relu1_1U1,3);
labels = Input('labels');
loss = vl_nnloss(output, labels, 'loss', 'mse');

Layer.workspaceNames();

net = Net(loss);


[net, info] = sparseNN_train(net, imdb, getBatch(opts,net.meta) ,opts) ;
% system('shutdown -s')


end




function fn = getBatch(opts,meta,gpu)
fn = @(x,y,z) getDagNNBatchSR(x,y,z) ;
end

function inputs = getDagNNBatchSR(imdb, batch, gpu)
% -------------------------------------------------------------------------
    
    % returns a batch of images or patches for training 
   
    images =  imdb.images.data(:,:,:,batch) ; % selects the correct batch 
	labels =  imdb.images.labels(:,:,:,batch) ; 
   
    
    images(:,:,1:3,:) = single(images(:,:,1:3,:))/255;% normalize batch to [0,1]
    images(:,:,4,:) = single(images(:,:,4,:))/80;


    labels = single(labels);

    if gpu 
        inputs = {'images',gpuArray(single(images(:,:,1:4,:))),'labels',gpuArray(single(labels))} ;
    else
        inputs = {'images',single(images(:,:,1:4,:)),'labels',single(labels)} ; %mac
    end 

end


function net_out = add_(net, opts, sz, order, varargin)
opts.weightInitMethod = 'morph';
% opts.cudnnWorkspaceLimit = 1024*1024*1204*4 ; % 1GB
% opts.batchNormalization = false ;

filters = Param('value', init_weight(opts, sz, 'single'), 'learningRate', 10^1); % 1  0
biases = Param('value', zeros(sz(4), 1, 'single'), 'learningRate', 10^3); % 3  0


net_top = net.^(order);
net_bot = net.^(order-1);

net_top1 = vl_nnconv(net_top, filters, biases, varargin{:}) ; % set the learning rate to 0
net_bot1 = vl_nnconv(net_bot, filters, biases, varargin{:}) ;

net_out = net_top1./net_bot1;

% "A Fast Thresholded Linear Convolution Representation of Morphological Operations"

% net = vl_nnconv(net, filters, biases, varargin{:}) ;
% net = net.*[net>0.5];

% net = vl_nnrelu(net) ;
end

function weights = init_weight(opts, sz, type)  %initialize the weight of filter (learning path) 

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(sz, type)*sc;
  case 'xavier'
    sc = sqrt(3/(sz(1)*sz(2)*sz(3))) ; 
    weights = abs( (rand(sz, type)*2 - 1)*sc ) ;   
  case 'morph'        
       
%     weights = (rand(sz,type))*(10^0); % 0 % initialize everything around zero
%     weights = 10^0*single(randn(sz,type)>0); % 0 % initialize everything around zero
       
%        weights = single(getnhood(strel('diamond',floor(sz(1)/2))));
%      weights = zeros(sz, type) ;    
%      weights(ceil(sz(1)/2),ceil(sz(2)/2)) = 1;
%      weights = weights / sum(weights(:));
     
%     sc = sqrt(3/(sz(1)*sz(2)*sz(3))) ; 
%     weights = (rand(sz, type)*2 - 1)*sc ;
    weights = ones(sz, type) ;

%     weights = ones(sz, type) ;
%     for i=1:sz(4)
%         kern = single(getnhood(strel('diamond',round(rand()*floor(sz(1)/2)))));
%         kern = padarray(kern,[(sz(1)-size(kern,1))/2 , (sz(2)-size(kern,2))/2]);
%         weights(:,:,:,i) = kern+10^-3;
%     end
	
    
    
    
  case 'xavierimproved'
    sc = sqrt(2/(sz(1)*sz(2)*sz(4))) ;  
    weights = randn(sz, type)*sc ;
%   otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end
end


function bw = morph_diamond(x_input, k)
    % x_input: the input of ; packed binary image of any dimension.

    % k: the morph. kernel size;
    r = floor(k/2);
    se = strel('diamond',r);
    bw = imdilate(x_input, se);
end 


