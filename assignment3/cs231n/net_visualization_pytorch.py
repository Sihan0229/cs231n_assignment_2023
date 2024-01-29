import torch
import random
import torchvision.transforms as T
import numpy as np
from .image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from scipy.ndimage.filters import gaussian_filter1d

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    # "test" mode下，模型通常会关闭一些训练中使用的特定层（如 dropout 或 batch normalization）
    model.eval()

    # Make input tensor require gradient
    # 告诉 PyTorch 在计算模型的前向传播时跟踪输入图像 X 的梯度。
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 执行Forward pass
    scores = model(X)
    # y.view(-1, 1) 将标签 y 变形为列向量
    # 使用 gather(1,) 从 scores 中选择正确类别的分数
    correct_scores = scores.gather(1, y.view(-1, 1))
    # Compute loss
    # 使用负的正确类别分数的和作为损失，因为我们想要最大化这个分数。
    loss = -correct_scores.sum()

    # 执行Backward pass
    loss.backward()
    # Compute the saliency map
    # 梯度.绝对值.三通道当中的最大值，dim=1即对应(N, 3, H, W)的3
    # 注意若没有[0]，则第一个张量是最大值的张量，第二个张量是对应最大值的索引。
    saliency = X.grad.abs().max(dim=1)[0]
    
    # Clear gradients for next iteration
    # 清零输入图像的梯度，以确保下一次迭代时梯度不会累积。
    X.grad.data.zero_()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image, and make it require gradient
    X_fooling = X.clone()
    X_fooling = X_fooling.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. You should perform gradient ascent on the score of the #
    # target class, stopping when the model is fooled.                           #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Set the model to evaluation mode
    model.eval()

    # Define a criterion (loss function) to maximize the target class score
    # criterion = torch.nn.CrossEntropyLoss()

    #初始分类
    scores = model(X_fooling)
    
    _, y_predit = scores.max(dim = 1)
    
    iter = 0
    
    while(y_predit != target_y):
        iter += 1
        
        target_score = scores[0, target_y]
        target_score.backward()
        grad = X_fooling.grad / X_fooling.grad.norm()
        X_fooling.data += learning_rate * grad
        
        X_fooling.grad.zero_()
        
        model.zero_grad()
        
        scores = model(X_fooling)
        _,y_predit=scores.max(dim = 1)

    print("Iteration Count: %d"% iter)
        

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling

def class_visualization_update_step(img, model, target_y, l2_reg, learning_rate):
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    ########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score = model(img)
    target_score = score[0,target_y]
    target_score.backward()
    
    im_grad = img.grad - l2_reg * img
    grad = im_grad / im_grad.norm()
    img.data += learning_rate * grad
    img.grad.zero_()
    model.zero_grad()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################


def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy() # 转换成numpy数组
    X_np = gaussian_filter1d(X_np, sigma, axis=2) # 水平方向
    X_np = gaussian_filter1d(X_np, sigma, axis=3) # 垂直方向
    X.copy_(torch.Tensor(X_np).type_as(X)) # 转换为pytorch张量
    return X

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    # 如果 ox 不等于零，表示需要在图像的水平方向进行抖动，
    # 将图像沿着水平方向切分成两部分，然后将右侧的部分移动到左侧。
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    # 如果 oy 不等于零，表示需要在图像的垂直方向进行抖动，
    # 将图像沿着垂直方向切分成两部分，然后将上侧的部分移动到下侧。 
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X
