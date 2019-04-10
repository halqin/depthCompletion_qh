% comparing two trained model
num_im = 100;
[m1,in1,m2,in2] = eval2model_path();

error1 = evalmodel.evalModel(in1,m1, num_im, 'mse');
% error2 = evalmodel.evalModel(in2,m2, num_im, 'mae');

% sprintf('The error of aniso is %f\nThe error of morph is %f', ...
%     sum(error1)/num_im, sum(error2)/num_im)

plot(error1);
% hold on
% plot(error2);
% hold off