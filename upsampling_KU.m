function veloDepthKu = upsampling_KU(veloDepth)
% veloDepth = imread('/scratch/mdimitri/depth_completion/depth_selection/val_selection_cropped/velodyne_raw_tiff/2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.tiff');
% veloDepth = single(veloDepth) / 256.0;

veloDepthInverted = [100 - veloDepth] .* single(veloDepth~=0);

veloDepthDilated = imdilate(veloDepthInverted,single(getnhood(strel('diamond',2)))); % diamond 5x5
veloDepthClosed = imclose(veloDepthDilated,ones([5,5]));

veloDepthFilled = imdilate(veloDepthClosed,ones([7,7]));
veloDepthFilled = veloDepthClosed + [veloDepthClosed==0].*veloDepthFilled; % fill in only holes

veloDepthFilled2 = imdilate(veloDepthFilled,ones([31,31])); % diamond [51,51],[25,25]
veloDepthFilled2 = veloDepthFilled + [veloDepthFilled==0].*veloDepthFilled2; % fill in only holes

veloDepthKu = 100 - veloDepthFilled2;
veloDepthKu = veloDepthKu .* [veloDepthKu~=100];
end



% veloDepthKu = veloDepth
% 
% veloDepthInverted = [100 - veloDepth] .* float(veloDepth!=0)
% 
% veloDepthDilated = imdilate(veloDepthInverted,strel_circ(2),[2,2]) % diamond 5x5
% veloDepthClosed = imclose(veloDepthDilated,ones([5,5]),[2,2])
% 
% veloDepthFilled = imdilate_gray(veloDepthClosed,ones([7,7]),[3,3]) % diamond 7x7
% veloDepthFilled = veloDepthClosed + [veloDepthClosed==0].*veloDepthFilled % fill in only holes
% 
% veloDepthFilled2 = imdilate_gray(veloDepthFilled,ones([31,31]),[15,15]) % diamond [51,51],[25,25]
% veloDepthFilled2 = veloDepthFilled + [veloDepthFilled==0].*veloDepthFilled2 % fill in only holes
% 
% veloDepthKu = 100 - veloDepthFilled2
% veloDepthKu = veloDepthKu .* [veloDepthKu!=100] 

