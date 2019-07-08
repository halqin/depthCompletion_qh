function [error_cnn, cnn_out]= cnnOuterror(input_data, input_labels, net)
% input_data RGB-D data
% input_labels: ground truth data
% net: the net object 
    if net.gpu
             net.eval({'images',gpuArray(single( input_data)), 'labels', gpuArray(single(input_labels))},'test');
    else
             net.eval({'images', input_data, 'labels', input_labels},'test');
    end
    error_cnn = gather(net.getValue('loss1'));
    try
        cnn_out = gather(net.getValue('prediction'));
    catch  
        cnn_out = gather(net.getValue('output'));
    end 
end