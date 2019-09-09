function [imidb] = generate_all_imdb()

dbpath = 'D:\depthdata\manual_all\';
train_size = 500;
    imdb = []; % this data structure is used during training and validation
    
    rng(2);
    close all

    valset = 0.01*1000; % first 116 images will be used for validation
    
    pathimage = ['D:\depthdata\manual_test\rgb\2011_09_26_drive_0001_sync\image_02\data\']; %RGB images 
    pathd = ['D:\depthdata\manual_test\train\2011_09_26_drive_0001_sync\proj_depth\velodyne_raw\image_02\']; 
    patgt = ['D:\depthdata\manual_test\train\2011_09_26_drive_0001_sync\proj_depth\groundtruth\image_02\']; % ground truth images 
    
    sz = [384,1280]/1; 
        
    d1 = dir([pathimage '*.png']);
    d2 = dir([pathd '*.png']);
    d3 = dir([patgt '*.png']);
    
    
       
    seqSize = size(d1,1);
     
     
    imdb.images.data = zeros([sz(1) sz(2) 4 seqSize],'single'); % RGBD 
    imdb.images.labels = zeros([sz(1) sz(2) 1 seqSize],'single'); % the ground truth data
	imdb.images.set = zeros([seqSize,1],'single');  % vector containig training and validation flags
    



end