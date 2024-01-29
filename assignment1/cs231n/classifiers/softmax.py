from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    for i in range(N):
        score = X[i].dot(W)
        exp_score = np.exp(score - np.max(score))
        loss += -np.log(exp_score[y[i]]/np.sum(exp_score)) / N
        #loss += (-np.log(exp_score[y[i]])+ np.log(np.sum(exp_score))) / N
        dexp_score = np.zeros_like(exp_score)
        dexp_score[y[i]] -= 1/exp_score[y[i]]/N
        dexp_score += 1 /np.sum(exp_score) / N
        dscore = dexp_score *exp_score
        dW += X[[i]].T.dot([dscore])
    loss +=reg*np.sum(W**2)
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
   
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    #exp_score = np.exp(score - np.max(score))
    scores -= np.max(scores, axis=1, keepdims=True)#保持dim
    exp_scores = np.exp(scores)

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute the loss
    N = X.shape[0]  #有点不熟悉这个维度012的顺序
    loss = np.sum(-np.log(probs[np.arange(N), y])) / N
    loss +=  reg * np.sum(W * W) #正则化强度的系数其实无所谓？只要不太小应该效果都差不多

    # Compute the gradient
    dscores = probs
    dscores[np.arange(N), y] -= 1
    dscores /= N

    dW = X.T.dot(dscores)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
