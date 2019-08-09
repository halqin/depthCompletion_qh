
base_lr = 0.000001;
max_lr = 0.002;

epoch = 40;
batchsize = 3;
num_im  = 461;
%iteration_epoch = num_im/batchsize; 
%iteration_all = iteration_epoch*epoch;
%iteration_count = 0; 
y=[];
for j = 1:epoch
    iteration_epoch = num_im/batchsize; 
    stepsize = iteration_epoch*10;
    x = batchprocess(batchsize, num_im, base_lr, max_lr,stepsize, j,iteration_epoch);
    y = [y,x]; 
end 

% for i = 1:iteration_all
%     x(i) = cycle_lr(i, stepsize, base_lr, max_lr); 
% end


plot(y);

function lr =  cycle_lr(iteration, stepsize, base_lr, max_lr)
    cycle = floor(1+ iteration/(2*stepsize));
     x = abs(iteration/stepsize - 2 * cycle + 1);
     lr = base_lr + (max_lr - base_lr) * max(0, (1-x)); 
end



function x = batchprocess(batchsize, num_im, base_lr, max_lr, stepsize, j, iteration_epoch)
    %persistent x, iteration_count
    count_  = 0;
    for i = 1:batchsize:num_im
        count_ = count_+1;
        iteration_count = iteration_epoch*(j-1) + count_; 
            x(count_)= cycle_lr(iteration_count, stepsize, base_lr, max_lr); 
    end
end 


