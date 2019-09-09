aa = load('F:\convnet\model_result\models\FusionA\demo_FusionA_sparse_layerout\layer_error.mat');
bb = load('F:\convnet\model_result\models\layer_error\layer_error.mat');
plot(aa.layer_loss);
hold on;
plot(bb.layer_loss);

lgd = legend('Without mask', 'With mask');
lgd.FontSize = 12;
% legend('Sparse', 'Morph', 'Aniso+Mask');
ts = title('Learning curve for different setting schedule');
ts.FontSize = 12; 
xlabel('Epochs');
ylabel('Loss');
grid on;
set(0,'DefaultLineLineWidth',2);