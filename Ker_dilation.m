clc;
close all;
imdb = [];
imdbGD= imread('GD02.png');
imdbrgb = imread('rgb02.png');
imdbraw = imread('raw02.png');

% Morphological structuring element
se = strel('diamond',2);
bw = imdilate(imdbraw, se);
se2 = strel('rectangle',[5 5]);
bw2 = imdilate(imdbraw, se2);
se3 = cross(5);
bw3 = imdilate(imdbraw, se3);

se4= circle_(5);
bw4 = imdilate(imdbraw, se4);


figure;
subplot(5,1,1);
imagesc(imdbraw);
title('Original');

subplot(5,1,2);
imagesc(bw);
title('diamond');

subplot(5,1,3);
imagesc(bw2);
title('full');

subplot(5,1,4);
imagesc(bw3);
title('Cross');

subplot(5,1,5);
imagesc(bw4);
title('circle');


bw = imclose(bw, se2);
bw2 = imclose(bw2, se2);
bw3 = imclose(bw3, se2);
bw4 = imclose(bw4, se2);



figure;
subplot(5,1,1);
imagesc(imdbraw);
title('Original');

subplot(5,1,2);
imagesc(bw);
title('diamond');

subplot(5,1,3);
imagesc(bw2);
title('full');

subplot(5,1,4);
imagesc(bw3);
title('Cross');

subplot(5,1,5);
imagesc(bw4);
title('circle');


fill_ = strel('rectangle',[7 7]);

bw = imclose(bw, fill_);
bw2 = imclose(bw2, fill_);
bw3 = imclose(bw3, fill_);
bw4 = imclose(bw4, fill_);



figure;
subplot(5,1,1);
imagesc(imdbraw);
title('Original');

subplot(5,1,2);
imagesc(bw);
title('diamond');

subplot(5,1,3);
imagesc(bw2);
title('full');

subplot(5,1,4);
imagesc(bw3);
title('Cross');

subplot(5,1,5);
imagesc(bw4);
title('circle');


fill_large = strel('rectangle',[35 35]);

bw = imclose(bw, fill_large);
bw2 = imclose(bw2, fill_large);
bw3 = imclose(bw3, fill_large);
bw4 = imclose(bw4, fill_large);



figure;
subplot(5,1,1);
imagesc(imdbraw);
title('Original');

subplot(5,1,2);
imagesc(bw);
title('diamond');

subplot(5,1,3);
imagesc(bw2);
title('full');

subplot(5,1,4);
imagesc(bw3);
title('Cross');

subplot(5,1,5);
imagesc(bw4);
title('circle');



bw = imgaussfilt(bw);
bw2 = imgaussfilt(bw2);
bw3 = imgaussfilt(bw3);
bw4 = imgaussfilt(bw4);





figure;
subplot(5,1,1);
imagesc(imdbraw);
title('Original');

subplot(5,1,2);
imagesc(bw);
title('diamond');

subplot(5,1,3);
imagesc(bw2);
title('full');

subplot(5,1,4);
imagesc(bw3);
title('Cross');

subplot(5,1,5);
imagesc(bw4);
title('circle');





function x = cross(r)
    x = zeros(r,r);
    x(3,:) = ones(1,r);
    x(:,3) = ones(1,r);
   
end 


function x = circle_(r)
    x = ones(r,r);
    x(1,1) = 0; 
    x(1,r) = 0;
    x(r,1) = 0 ;
    x(r,r) = 0;
end
