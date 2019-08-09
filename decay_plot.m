
base_lr = 0.000001;
max_lr = 0.002;

epoch = 40;
batchsize = 3;
num_im  = 461;
stepsize = 4; 
num_iter = ceil (((num_im/batchsize)*epoch)/stepsize); 
cunt = 0;
x  = step_decay(max_lr, base_lr, stepsize);
% x(5) = 1e-07; 
% x= step_decay(epoch, max_lr, base_lr);
% cunt = 0;
for i=1:epoch
    %x= step_decay(epoch, max_lr, base_lr);
    if i<epoch/4
            lr = x(1);
    elseif  i<(epoch/4)*2
            lr = x(2);
    elseif  i<(epoch/4)*3
            lr = x(3);
    else 
            lr = x(4);
    end
    
    for j = 1:batchsize:num_im
      cunt = cunt+1;
        y(cunt) = lr; 
      
    end
end

plot(y);

function lr_cell = step_decay(maxlr, minlr, stepsize)
%num_epoch: the number of epoches
%lr_cell: the learning rate list
for i = 1:stepsize
    lr_step = (maxlr-minlr)/stepsize;
    lr_cell(i) = maxlr-lr_step*i;
end
end