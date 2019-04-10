function input = fixFilter(input, filter)
%input = imdb.images.data(:,:,4,i)
%filter: bilteral; imdiff
    switch filter
        case 'bilteral'
            input =80*( imbilatfilt(input/80,'DegreeOfSmoothing', 12, 'SpatialSigma', 2));
        case 'imdiff'
            input = 80*( imdiffusefilt(input/80, 'GradientThreshold', 14, 'NumberOfIterations', 3, 'Connectivity', 'minimal'));                     
    end   
end 