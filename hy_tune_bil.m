% tuneing the hyperparameter of bilateral filter 
[input_name,save_name] = hy_tune_bil_path();
load(input_name);


hy_len = 100; % the length of the hyperparameter vector
val_im = 100; % the number images used for validation 
error_sum = 0;
ave_error = 0;

sigma_vec = 1+(10-1)* rand(hy_len,1); %DegreeOfSmoothing
% r = a + (b-a).*rand(N,1) -----> (a,b) 
smooth_vec = 1+(20-1)* rand(hy_len,1); % spatialSigma

imdb.images.data(:,:,4,:) = imdb.images.data(:,:,4,:)/80;
imdb.images.data(:,:,1:3,:) = imdb.images.data(:,:,1:3,:)/255;

for j = 1:hy_len
    sigma = sigma_vec(j,1);
    smooth = smooth_vec(j,1);
    for i = 1:val_im
         imdb.images.data(:,:,4,i) = imbilatfilt(imdb.images.data(:,:,4,i),'DegreeOfSmoothing', smooth, 'SpatialSigma', sigma);
         error_in = input_error(imdb.images.data(:,:,4,i), imdb.images.labels(:,:,1,i));
         error_sum = error_sum +  error_in;
    end 
     ave_error = [ave_error error_sum/val_im];
     error_sum = 0;
end 






function error_in = input_error(input_data, input_labels )
%          input_data = imdb.images.data(:,:,4,i) ;
%          input_labels =  imdb.images.labels(:,:,1,i));
         instanceWeights = input_labels ~= 0 ;
         
         t = (80* input_data- input_labels) .^ 2 ;
         y = instanceWeights .*t;
         y = sum(y);
         error_in = y/sum(sum(instanceWeights));
         error_in = sqrt(sum(error_in));
         
end