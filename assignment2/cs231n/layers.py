from builtins import range
from anyio import BrokenResourceError
import numpy as np
import torch
import torch.nn.functional as nnf



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_vector = x.reshape(x.shape[0], -1)
    out = x_vector.dot(w)
    out += b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # shape ：x(10*2*3) w(6*5) b(5) out(10*5)
    dx = np.dot(dout, w.T)  # (N, M) dot (D, M).T -> (N, D): (10, 6)
    dx = dx.reshape(x.shape)  # 将 dx 调整为与 x 相同的形状: (10*2*3)

    # 计算关于 w 的梯度
    dw = np.dot(x.reshape(x.shape[0], -1).T, dout)  # (D, N) dot (N, M) -> (D, M) 6*10 dot 10*5 = 6*5

    # 计算关于 b 的梯度
    db = np.sum(dout, axis=0)  # 沿着样本维度求和，得到 (M,) 形状的梯度


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(x,0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    dx = np.copy(dout)  
    dx[x <= 0] = 0


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

         # 均值和方差
        sample_mean = np.mean(x, axis=0)  # (N, D) --> (D,)
        sample_var = np.var(x, axis=0)  # (D,)

        # 使用样本均值和方差进行归一化
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)  # (N, D)

        # 缩放和平移(线性变换)：为了恢复数据本身的表达能力
        out = gamma * x_hat + beta  # (N, D)

        cache = (x, gamma, beta, x_hat, sample_mean, sample_var, eps)

        # 基于momentum的指数衰减，从而估计出运行测试集时的均值和方差
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean  # (D,)
        running_var = momentum * running_var + (1 - momentum) * sample_var  # (D,)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/ MODIFY THIS LINE)*****

        pass
        # 使用train的均值和方差进行归一化
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)  # (N, D)
        # 缩放和平移
        out = gamma * x_hat + beta  # (N, D)
        cache = (x, gamma, beta, x_hat, running_mean, running_var, eps)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    x, gamma, beta, x_hat, mean, var, eps = cache

    N, D = dout.shape

    # some para settings
    xmean = x - mean
    sqrtvar = np.sqrt(var + eps)
    ivar = 1. / sqrtvar

    # step9: out = gammax_hat + beta
    dbeta = np.sum(dout, axis=0)
    dgammax_hat = dout  # not necessary, but more understandable
    
    # step8: gammax_hat = gamma * x_hat
    dgamma = np.sum(dgammax_hat * x_hat, axis=0)
    dx_hat = dgammax_hat * gamma

    # step7: x_hat = xmean * ivar
    divar = np.sum(dx_hat * xmean, axis=0)
    dxmean1 = dx_hat * ivar

    # step6: ivar = 1 / sqrtvar
    dsqrtvar = -1. / (sqrtvar**2) * divar

    # step5: sqrtvar = sqrt(var + eps)
    dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar

    # step4: var = 1 / N * sum(sq**2)
    dsq = 1. / N * np.ones((N, D)) * dvar

    # step3: sq = xmean**2
    dxmean2 = 2 * xmean * dsq
    # step2: xmean = x - mean
    dx1 = (dxmean1 + dxmean2)
    dmean = -1 * np.sum(dxmean1 + dxmean2, axis=0)
    # step1: mean = 1 / N * sum(x)
    dx2 = 1. / N * np.ones((N, D)) * dmean
    # step0: x
    dx = dx1 + dx2


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    x, gamma, beta, x_hat, mean, var, eps = cache
    N, D = dout.shape

    # 针对beta和gamma的梯度
    dbeta = np.sum(dout, axis=0)  # (D,)
    dgamma = np.sum(dout * x_hat, axis=0)  # (D,)

    # 针对x的梯度
    dx_hat = dout * gamma
    dvar = np.sum(dx_hat * (x - mean) * (-0.5) * np.power(var + eps, -1.5), axis=0)  # (D,)
    dmean = np.sum(dx_hat * (-1) / np.sqrt(var + eps), axis=0) + dvar * np.sum(-2 * (x - mean), axis=0) / N  # (D,)
    dx = dx_hat / np.sqrt(var + eps) + dvar * 2 * (x - mean) / N + dmean / N  # (N, D)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    # layer norm在每个样本上计算，不需要区分train和test，因此不需要计算running average.
    N, D = x.shape
    # 计算样本均值和方差
    sample_mean = np.mean(x, axis=1).reshape(N, 1)  # (N, D) --> (N,)
    sample_var = np.var(x, axis=1).reshape(N, 1)  # (N,)
    # 使用样本均值和方差进行归一化
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)  # (N, D)
    # 缩放和平移(线性变换)：为了恢复数据本身的表达能力
    out = gamma * x_hat + beta  # (N, D)
    cache = (x, gamma, beta, x_hat, sample_mean, sample_var, eps)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    # 参考BatchNormalization.ipynb中的反向传播公式
    x, gamma, beta, x_hat, mean, var, eps = cache
    N, D = dout.shape

    # 针对beta和gamma的梯度
    dbeta = np.sum(dout, axis=0)  # (D,)
    dgamma = np.sum(dout * x_hat, axis=0)  # (D,)

    # 针对x的梯度: 注意reshape变形为(N, 1)!!!
    dx_hat = dout * gamma
    dvar = np.sum(dx_hat * (x - mean) * (-0.5) * np.power(var + eps, -1.5), axis=1).reshape(N, 1)  # (N, 1)
    dmean = np.sum(dx_hat * (-1) / np.sqrt(var + eps), axis=1).reshape(N, 1) + dvar * np.sum(-2 * (x - mean), axis=1).reshape(N, 1) / D  # (N, 1)
    dx = dx_hat / np.sqrt(var + eps) + dvar * 2 * (x - mean) / D + dmean / D  # (N, D)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.注意是保持or丢弃？
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param: #如果dropout_param中包含了种子seed，则通过np.random.seed()设置了随机数生成的种子。
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        # 使用inverted dropout: 在训练的时候drop和调整数值范围，测试时不做任何事.
        # 生成mask
        mask = (np.random.rand(*x.shape) < p) / p  # (N, D),为了保持期望值不变，还将掩码除以p。
        # 计算out
        out = x * mask  # (N, D)
        # 感觉还是不建议的那个方法
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        out = x 
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache

def dropout_forward_2(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.注意是保持or丢弃？
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param: #如果dropout_param中包含了种子seed，则通过np.random.seed()设置了随机数生成的种子。
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        # 使用inverted dropout: 在训练的时候drop和调整数值范围，测试时不做任何事.
        # 生成mask
        mask = (np.random.rand(*x.shape) < p)   # (N, D),为了保持期望值不变，还将掩码除以p。
        # 计算out
        out = x * mask/p  # (N, D)
        #out_without_p = x * mask
        # 感觉还是不建议的那个方法
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        out = x
        #out_without_p = x
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    #out_without_p  = out_without_p .astype(x.dtype, copy=False)

    return out,cache#,out_without_p

def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        dx = dout * mask  # (N, D)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    # x: Input data of shape (N, C, H, W)
    N, C, H, W = x.shape
    # w: Filter weights of shape (F, C, HH, WW)
    F, _, HH, WW = w.shape # C 已经赋值了所以"_"

    stride, pad = conv_param["stride"], conv_param["pad"]

    # 用课上的公式计算输出的尺寸
    H_out = 1 + (H + 2 * pad - HH) // stride 
    W_out = 1 + (W + 2 * pad - WW) // stride

    # 初始化输出
    out = np.zeros((N, F, H_out, W_out))

    # 在每个样本的每个通道的高和宽的上下都填充pad个0
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode="constant", constant_values=0)  # (N, C, H+2*pad, W+2*pad)
    
    # 卷积
    for n in range(N):  # 遍历样本
        for f in range(F):  # 遍历卷积核
            for i in range(H_out):  # 遍历输出的高
                for j in range(W_out):  # 遍历输出的宽
                    # 计算卷积：第n个sample与第f个filter在out的第i行第j列的输出
                    out[n, f, i, j] = np.sum(x_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] * w[f, :, :, :]) + b[f]
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache

def conv_forward_torch(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape # C 已经赋值了所以"_"

    stride, pad = conv_param["stride"], conv_param["pad"]

    # Transition numpy arrays to torch tensors
    x_torch = torch.from_numpy(x).float()
    w_torch = torch.from_numpy(w).float()
    b_torch = torch.from_numpy(b).float()

    # Perform the convolution operation using torch.nn.functional.conv2d
    out = nnf.conv2d(x_torch, w_torch, bias=b_torch, stride=stride, padding=pad)

    # Convert the output tensor back to numpy array
    out = out.data.numpy()

    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
   
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]
    _, _, H_out, W_out = dout.shape

    # 初始化梯度
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode="constant", constant_values=0)  # (N, C, H+2*pad, W+2*pad)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # 计算梯度
    for n in range(N):
        for f in range(F):
            db[f] += np.sum(dout[n, f, :, :])
            for i in range(H_out):
                for j in range(W_out):
                    dw[f, :, :, :] += x_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] * dout[n, f, i, j]
                    dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += w[f, :, :, :] * dout[n, f, i, j]
    # 去除padding
    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]  # (N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]

    # 计算输出的尺寸
    H_out = (H - pool_height) // stride + 1
    W_out = (W - pool_width) // stride + 1
    # 检查stride和pool_height/width是否兼容
    #assert pool_height % stride == 0, "Stride and pool_height are not compatible"
    #assert pool_width % stride == 0, "Stride and pool_width are not compatible"
    H_out = int(H_out)
    W_out = int(W_out)

    # 初始化输出
    out = np.zeros((N, C, H_out, W_out))
    # 池化
    for n in range(N):  # 遍历样本
        for c in range(C):  # 遍历通道
            for i in range(H_out):  # 遍历输出的高
                for j in range(W_out):  # 遍历输出的宽
                    # 计算池化：第n个sample的第c个channel在out的第i行第j列的输出
                    out[n, c, i, j] = np.max(x[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    x, pool_param = cache
    N, C, H_out, W_out = dout.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    # 初始化梯度
    dx = np.zeros_like(x)
    # 计算梯度
    for n in range(N):  # 遍历样本
        for c in range(C):  # 遍历通道
            for i in range(H_out):
                for j in range(W_out):
                    idx_max = np.argmax(x[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width])  # 找到最大值的索引
                    #这里相当于是在一块池化的范围内确定选的是哪一个
                    # 比如对8*8的x做pool大小为2*2的池化
                    # 这里确定的是每个2*2池化块里究竟选的是0-3哪一个
                    # 行列都是从0开始数的
                    # 而并非确定是8*8当中的哪一个位置
                    h_max, w_max = np.unravel_index(idx_max, (pool_height, pool_width))  # 将一维索引转换为二维索引
                    dx[n, c, i*stride+h_max, j*stride+w_max] += dout[n, c, i, j]  # 只将梯度传递给最大值所在的位置
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 通过转换维度调用vanilla版本的batch normalization
    # 之前的D是这里的C，之前的N在这里则是N*H*W。
    N, C, H, W = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(-1, C)  # (N, C, H, W) --> (N, H, W, C) --> (N*H*W, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # (N*H*W, C) --> (N, H, W, C) --> (N, C, H, W)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W=dout.shape
    dout_new=dout.transpose(0, 2, 3, 1).reshape(-1, C) 
    dx, dgamma, dbeta = batchnorm_backward_alt(dout_new,cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2) 
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    N, C, H, W = x.shape
    x_group = x.reshape(N, G, C//G, H, W)  # (N, C, H, W) --> (N, G, C//G, H, W)
    # 计算样本均值和方差
    sample_mean = np.mean(x_group, axis=(2, 3, 4), keepdims=True)  # (N, G, 1, 1, 1)
    sample_var = np.var(x_group, axis=(2, 3, 4), keepdims=True)  # (N, G, 1, 1, 1)
    # 使用样本均值和方差进行归一化
    x_hat = (x_group - sample_mean) / np.sqrt(sample_var + eps)  # (N, G, C//G, H, W)
    x_hat = x_hat.reshape(N, C, H, W)
    # 缩放和平移
    out = gamma * x_hat + beta  # (N, C, H, W)
    cache = (x, gamma, beta, G, x_hat, sample_mean, sample_var, eps)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    # 与layernorm_backward类似
    x, gamma, beta, G, x_hat, mean, var, eps = cache
    N, C, H, W = dout.shape

    # 针对beta和gamma的梯度
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)

    # 针对x的梯度
    x_group = x.reshape(N, G, C//G, H, W)  # (N, C, H, W) --> (N, G, C//G, H, W)
    dx_hat = (dout * gamma).reshape(N, G, C//G, H, W)  # (N, G, C//G, H, W)
    dvar = np.sum(dx_hat * (x_group - mean) * (-0.5) * np.power(var + eps, -1.5), axis=(2, 3, 4), keepdims=True)  # (N, G, 1, 1, 1)
    D = C//G * H * W
    dmean = np.sum(dx_hat * (-1) / np.sqrt(var + eps), axis=(2, 3, 4), keepdims=True) + dvar * np.sum(-2 * (x_group - mean), axis=(2, 3, 4), keepdims=True) / D  # (N, G, 1, 1, 1)
    dx = dx_hat / np.sqrt(var + eps) + dvar * 2 * (x_group - mean) / D + dmean / D  # (N, G, C//G, H, W)
    dx = dx.reshape(N, C, H, W)  # (N, G, C//G, H, W) --> (N, C, H, W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = x.shape[0]
    scores = x - np.max(x, axis=1, keepdims=True)
    correct_class_scores = scores[np.arange(num_train), y]
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins) / num_train

    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(num_train), y] -= num_pos
    dx /= num_train
        

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = x.shape[0]
    scores = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    #correct_class_scores = scores[np.arange(num_train), y]

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    loss = np.sum(-np.log(probs[np.arange(num_train ), y])) / num_train 

    # Compute the gradient
    dscores = probs
    dscores[np.arange(num_train ), y] -= 1
    dscores /= num_train 

    dx = dscores

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
