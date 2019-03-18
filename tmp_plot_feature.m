% tmp_plot feature
figure;
subplot(3,1,1);
aa_ = net.vars{313,1};
imagesc(aa_(:,:,15));
title('relu1_1U1');

subplot(3,1,2);
aa_ = net.vars{321,1};
imagesc(aa_);
title('conv28');

subplot(3,1,3);
aa_ = net.vars{323,1};
imagesc(aa_);
title('prediction');
                                                              
aa_ = net.vars{313,1}; 
aa_s = size(aa_);
figure;
for i =  1:aa_s(1,3)
    subplot(4,4,i);
    imagesc(aa_(:,:,i));
end 
    



% polt the input image annd ground truth:
figure;
subplot(1,2,1);
imagesc(imdb.images.data(:,:,4,44));
subplot(1,2,2);
imagesc(imdb.images.labels(:,:,1,44));

% check the error distribution:
% figure;
error=zeros(384,1280,100);
image_num = size(imdb.images.labels);
for i = 1:image_num(4)
    error(:,:,i) = abs(imdb.images.data(:,:,4,i) - imdb.images.labels(:,:,1,i));
end 
error_sum=sum(error,1);
error_sum = sum(error_sum,2);
vect  = reshape(error_sum, [1 100]);
bar(vect);


