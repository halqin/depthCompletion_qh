% [input_name,save_name] = hy_tune_bil_path();
input_name = '/Users/Hall/convnn/depthCompletionNet/imdb_sparse_100morph_test.mat';
load(input_name);


hy_len = 100; % the length of the hyperparameter vector
val_im = 100; % the number images used for validation 
error_sum = 0;
ave_error = 0;

grandTh_vec = 5+(15-5)* rand(hy_len,1); %GradientThreshold
numIt_vec = randi([1,15], hy_len, 1); %NumberOfIterations
con_vec=randi([1,2], hy_len, 1); %Connectivity
con_max_min = ["maximal", "minimal"]; 

imdb.images.data(:,:,4,:) = imdb.images.data(:,:,4,:)/80;
imdb.images.data(:,:,1:3,:) = imdb.images.data(:,:,1:3,:)/255;

for j = 1:hy_len
    grandTh = grandTh_vec(j);
    numIt = numIt_vec(j);
    con = con_max_min(con_vec(j));
    for i = 1:val_im
         imdb.images.data(:,:,4,i) = imdiffusefilt(imdb.images.data(:,:,4,i),'GradientThreshold', grandTh, 'NumberOfIterations', numIt, 'Connectivity',con);
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


function plot ()
    figure;
    x_axis = 2:(hy_len+1);
    plot(ave_error(2:101)); 
    title('anstropic difussion')
    xlabel('epoch');
    ylabel('error');
     [val, pos] = max(ave_error)


    grandTh_vec(pos-1)
    numIt_vec(pos-1)
    con_vec(pos-1)

     [val_min, pos_min] = min(ave_error(2:101))
     grandTh_vec(pos_min-1)
    numIt_vec(pos_min-1)
    con_vec(pos_min-1)

end 


