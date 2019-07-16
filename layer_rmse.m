% function [error_cnn, cnn_out]= layer_rmse(input_data, input_labels, net, )
% this function is used for checking the rmse in layers 
 setup_autonn;
 vl_setupnn;
num_im = 500;
[input_name, model_name,model_name2] = layer_rmse_path();

error = evalmodel.evalModel(input_name, model_name, num_im, 'mse', 'output');


% end 