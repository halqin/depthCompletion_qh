function [error_in, resi] = inputError(input_data, input_labels )
%input_data  is sparse with 1 channel; input data should be normalized between 0~1
%input_labels is sparse ground truth 
%resi: the error residual, could used for error map 
         instanceWeights = input_labels ~= 0 ;        
         t = (80* input_data- input_labels) .^ 2 ;
         y = instanceWeights .*t;
         resi = instanceWeights .*abs(80* input_data- input_labels);
         y = sum(y);
         error_in = gather( y/sum(sum(instanceWeights)));
         error_in = sqrt(sum(error_in));
end