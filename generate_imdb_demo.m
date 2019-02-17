function [imdb] = generate_imdb_demo()
% demo script for sampling the RAW kitti dataset

%dbpath = '/scratch/mdimitri/depth_completion/depth_selection/val_selection_cropped';
dbpath = 'D:\depthdata\data_depth_selection\depth_selection\val_selection_cropped';

    imdb = []; % this data structure is used during training and validation
    
    rng('shuffle');
    close all

    valset = 40; % first 116 images will be used for validation
    
    pathimage = [dbpath '\image\']; %RGB images 
    pathd = [ dbpath '\velodyne_raw\' ]; 
    patgt = [ dbpath '\groundtruth_depth\' ]; % ground truth images 
    
    sz = [384,1280]/1; 
    
    seqSize = 100; % for loading the 1000 val_selection_cropped images

    % initialize the memory
    imdb.images.data = zeros([sz(1) sz(2) 4 seqSize],'single'); % RGBD 
    imdb.images.labels = zeros([sz(1) sz(2) 1 seqSize],'single'); % the ground truth data
	imdb.images.set = zeros([seqSize,1],'single');  % vector containig training and validation flags
    
    
    d1 = dir([pathimage '*.png']);
    d2 = dir([pathd '*.png']);
    d3 = dir([patgt '*.png']);
    
    [Xq,Yq] = meshgrid([1:sz(2)],[1:sz(1)]);
    
    for i=1:size(d1,1)-900 % iterate all images in the folder       
        tic();        
        img   = imread([pathimage d1(i).name]);
        imgD  = imread([pathd d2(i).name]);
        imgGt = imread([patgt d3(i).name]);

        % pad images to 1280x384
        vertpad = (384 - size(img,1))/2;
        horpad  = (1280 - size(img,2))/2;        
        img   = pad(img,vertpad,horpad);
        imgD   = pad(imgD,vertpad,horpad);
        imgGt   = pad(imgGt,vertpad,horpad);
        
        imdb.images.data(:,:,1:3,i)   = single(img);  %1-3 channel is RGB image
        imdb.images.data(:,:,4,i)     = single(imgD);   % the forth channel is velodyne image
        
%         % basic interpolation to fix sparsity % 
%         [x,y,z] = find(imgD); x = double(x); y = double(y); z = double(z);
%         F = TriScatteredInterp(x,y,z,'nearest');  % natural % linear 
%         Zq = F(Yq,Xq);
% %         imagesc(Zq);
%         imdb.images.data(:,:,4,i)     = Zq;        
%         % basic interpolation to fix sparsity % 
        
        % interpolation from paper: In Defense of Classical Image Processing: Fast Depth Completion on the CPU Jason Ku, Ali Harakeh, Steven L. Waslander %
%         imdb.images.data(:,:,4,i)     = upsampling_KU(single(imgD)/256);        
        % interpolation from paper: In Defense of Classical Image Processing: Fast Depth Completion on the CPU Jason Ku, Ali Harakeh, Steven L. Waslander %
        
        imdb.images.labels(:,:,1,i)   = (single(imgGt)/256); %????????? why /256 as label        
        imdb.images.set(i) = i>valset;  % first 116 images will be used for validation
        elapsed = toc();
        fprintf('%d/%d %f seconds, %s \n',i,size(d1,1),elapsed,d1(i).name);
    end
    
    % save the matlab variable
    save('.\imdb_sparse.mat','imdb','-v7.3');
    
    end
    

                function out = pad(in,vertpad,horpad)
                    out = padarray(in,[floor(vertpad),floor(horpad)],0,'pre');
                    out = padarray(out,[ceil(vertpad),ceil(horpad)],0,'post');
                end
  
           
