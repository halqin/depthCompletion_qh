Introduce what I have done at this moment. 

* I have watched the cs231n. But I have not get a real project to implement. 
I have ordered a graphic card, but this is out of stock at this moment. 
Then I will try to use AWS. for students? 
* setting a clear target 


About the data: 
What is the annotated, selection, velodyne respectively? 
annotated -- label?	


* supervised learning, how about KNN 
* What is the label? THe color?
* There is only black and white 
---------------------------------------------

As promised here are some details to get you started on using MatConvNet for Matlab.

The main page and installation info can be found here:
http://www.vlfeat.org/matconvnet/#matconvnet-cnns-for-matlab

For experimenting I would strongly advise to also install the AutoNN project:
https://github.com/vlfeat/matconvnet-contrib
https://github.com/vlfeat/autonn

My depth completion code uses blocks from both MatConvNet and AutoNN:
https://github.com/mdimitri/depthCompletionNet

In the repository I've provided demo scripts for creating a data set "generate_imdb_demo.m"
which creates the data structure used to train the network.
"train_net_demo.m" creates a standard U-Net structure for training with the MSE loss function. 
It then calls "cnn_train_autonn_demo.m" which initializes the actual training. 
I suggest you get familiar with these scripts. Let me know if you run into trouble.

---------------------------------------------

what is the function of each script:
"generate_imdb_demo.m": creating a data set 
"train_net_demo.m" : creates a standard U-Net structure for training with the MSE loss function. 
"cnn_train_autonn_demo.m": initializes the actual training. 

---------------------------------------------




07/02/2019
* if I don't use all the data, the loss is too high. If I use all the data, the traning time is too long (2 days). 
* How much data should I use in this stage. 
* I did not improve the CNN framework yet, should I introduce the detail of the framwork? 


after meeting: 
overview: KNN, linear, traditional way 


_________________________________________________

02/26/10§9
I read the paper again. And do some experiment by using morphological kernel.   

* The network structure is defined in the train net. 
The function of cnn_train_autonn: 
- iteration epoch 
- train, calculate loss
- plot 

Then whats the different of simpleNN/ DagNN/ autoNN 


* How to plot output in each layer 
getfields not work 

* How the morpholocial layer connected with CNN. THe RV prediction connected to nonthing. 

add_ have more than 5 arguements 

check the loss function: 
k>> figure, imagesc(x(:,:,:,1));
K>> figure, imagesc(c(:,:,:,1));
K>> figure, imagesc(instanceWeights(:,:,:,1));


After the meeting: 
* if we want fully use the mem. in GPU, we could change the stripeSize and the batch size. 



% -----------------------------------------------------------
%-------------------------------------------------------------
% -----------------------------------------------------------
%-------------------------------------------------------------
% -----------------------------------------------------------
%-------------------------------------------------------------

* 必须增加mex路径到matlab,不然运行vl_conv 会报错

* The Stats class keeps track of values such as objectives and errors during training, and can be used to easily plot them.
时刻关注某些数值的变化.

stats = Stats({'objective', 'error'}) ;


两个参数分别带表两个名字
 S = Stats({'var1', 'var2', ...}) 

 -- creates a Stats object 
 -- registers variables with names 'var1', 'var2', etc.

 These are the variables from a network that need to be plotted, such as objectives and error metrics.

需要plot 什么就加什么名字. 这里需要plot loss, objective(目标).
最后就会plot出两个figure, 一个名字叫objective, 一个名字叫loss

stats.push 会把计算得到的数值push到stats.history
stats.plot 就会把stats.history 进行plot

stats.push('自己任意取名字') . 最后出的图就会用自己任意取的名字作为legend
stats.print() writes the current statistics to the terminal.

如果过想查看train 或者 val 的 error数值:
stats object. values('train','error')
如果过想查看train 或者 val 的 inference数值:
stats object. values('train', 'inference')





* loss category: 
 Classification error:: `classerror`
    L(X,c) = (argmax_q X(q) ~= c). Note that the classification
    error derivative is flat; therefore this loss is useful for
    assessment, but not for training a model.

  Top-K classification error:: `topkerror`
    L(X,c) = (rank X(c) in X <= K). The top rank is the one with
    highest score. For K=1, this is the same as the
    classification error. K is controlled by the `topK` option.

  Log loss:: `log`
    L(X,c) = - log(X(c)). This function assumes that X(c) is the
    predicted probability of class c (hence the vector X must be non
    negative and sum to one).

  Softmax log loss (multinomial logistic loss):: `softmaxlog`
    L(X,c) = - log(P(c)) where P(c) = exp(X(c)) / sum_q exp(X(q)).
    This is the same as the `log` loss, but renormalizes the
    predictions using the softmax function.

  Multiclass hinge loss:: `mhinge`
    L(X,c) = max{0, 1 - X(c)}. This function assumes that X(c) is
    the score margin for class c against the other classes.  See
    also the `mmhinge` loss below.

  Multiclass structured hinge loss:: `mshinge`
    L(X,c) = max{0, 1 - M(c)} where M(c) = X(c) - max_{q ~= c}
    X(q). This is the same as the `mhinge` loss, but computes the
    margin between the prediction scores first. This is also known
    the Crammer-Singer loss, an example of a structured prediction
    loss.



* SEQUENTIALNAMES 不用刻意的调用, 在compile network 的时候就会默认被调用

lternatively, loss.sequentialNames() would fill in unnamed layers involved in the computation of loss, using intuitive names like convN for the Nth convolutional layer, and convN_filters for the corresponding filters. To ensure that all layers have names, network compilation will call this function.



* 如何获取learning parameter/layer output? 

parameter:
net.getValue('conv2_filters') 
'conv2_filters' 的获取位置: net.params.name 
layer output:
net.getValue('layerName');

*查看函数的所有相关路径:
which -all vl_simplenn
which -all vl_nnconv


* 关于eval的使用方法:
In traning:
net.eval({'images',images, 'labels',labels})

In validation:
net.eval({'images',images, 'labels',labels}, 'test')

eval 说明文档:
OBJ.EVAL(INPUTS, MODE)
-- INPUTS = {'input1', value1, 'input2', value2, ...}
-- MODE
	'normal(default): forward + backward 
	test: only forward. set the testMode input to true 
	forward: only forward. without setting the testMode input to true 
	backward: only backward'
-- OBJ.EVAL(INPUTS, MODE, DEROUTPUT, false) : 在之前derivatives的基础上计算
-- OBJ.EVAL(INPUTS, MODE, DEROUTPUT): 不在之前derivative的基础上计算


* Logical Short-Circuiting
With logical short-circuiting, the second operand, expr2, is evaluated only when the result is not fully determined by the first operand, expr1.
For example, in the expression A && B, MATLAB® does not evaluate condition B at all if condition A is false. If A is false, then the value of B does not change the outcome of the operation.


* vl_nnloss
如果不加任何参数就是softmax:
  objective = vl_nnloss(x, labels) ;  % what we minimize
如果加了参数就可以是任何了:
  error = vl_nnloss(x, labels, 'loss', 'classerror') ;  % the error metric



* evalin
Execute MATLAB expression in specified workspace

* obj 使用例子:
opts.extractStatsFn = @extractStatsAutoNN ;
fn = opts.extractStatsFn ;
opts.extractStatsFn = @(stats, net, batchSize) fn(stats, net, sel, batchSize) ;
stats = params.extractStatsFn(stats, net, batchSize / max(1, numGpus)) ;
最终实际上是调用了extractStatsAutoNN


* structure 中取出整个field
KNN is a nested structure
使用[], 创建vector
aa= [KNN.stats.train.loss1]


* cell 中取出element
args is cell 
多个: args([1 2])
单个: args(2)












