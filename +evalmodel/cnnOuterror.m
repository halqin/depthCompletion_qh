function error_cnn = cnnOuterror(input_data, input_labels, net)
% input_data RGB-D data
% input_labels: ground truth data
% net: the net object 
    if net.gpu
             net.eval({'images',gpuArray( input_data), 'labels', gpuArray(single(input_labels))},'test');
    %          cnn_out = gather(net.getValue('prediction'));
    else
             net.eval({'images', input_data, 'labels', input_labels},'test');
    end
    error_cnn = gather(net.getValue('loss1'));
end