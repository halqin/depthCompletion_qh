% this script use the morphological mask to optimize the filtered image 
[morp_path, ani_path, save_name] = maskim_gener_path() ;

load(morp_path);
aa = load(ani_path);

size_= size(imdb.images.data);
% num_im = 100;
for i = 1:size_(4)
morplabel = imdb.images.labels(:,:,1,i);
morphin = imdb.images.data(:,:,4,i); 
aniin = aa.imdb.images.data(:,:,4,i) ;

imdb.images.data(:,:,4,i) = evalmodel.morph_mask(morphin, morplabel, aniin, 2);
end 

save(save_name, 'imdb', '-v7.3'); 

% 
% [e_final,resi_final] = evalmodel.inputError(morphin/80, morplabel);
% [e_ani,resi_ani] = evalmodel.inputError(aniin/80, morplabel);
