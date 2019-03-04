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
for i =  1:size_(3)
    subplot(4,4,i);
    imagesc(aa_(:,:,i));
end 
    