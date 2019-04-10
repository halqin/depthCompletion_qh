[input_name, save_name] = filterOut_path();
load(input_name)
size_= size(imdb.images.data);
for i = 1: size_(4)
    imdb.images.data(:,:,4,i) = depthmodel.fixFilter(imdb.images.data(:,:,4,i),'imdiff');
end 

save(save_name, 'imdb', '-v7.3'); 
