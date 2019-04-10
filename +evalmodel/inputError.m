function error_in = inputError(input_data, input_labels )
%input_data  is sparse with 1 channel
%input_labels is sparse ground truth 
         instanceWeights = input_labels ~= 0 ;
         
         t = (80* input_data- input_labels) .^ 2 ;
         y = instanceWeights .*t;
         y = sum(y);
         error_in = y/sum(sum(instanceWeights));
         error_in = sqrt(sum(error_in));
         
end