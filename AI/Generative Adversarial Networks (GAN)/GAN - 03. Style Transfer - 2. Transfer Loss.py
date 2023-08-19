# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F


# Defind a class for Gram Matrix
def gram_matrix(input):
    a, b, c, d = input.size()  # a: batch_size, b: num of maps, c: height, d: width

    # Get features
    #  : Transfer input tensor to 2d tensor -> Vectorize each feature map -> Combine into a single matrix
    features = input.view(a * b, c * d)

    # Calculate gram prediction
    #   : Calculate gram matrix using 2d feature map -> Calculate internal feature maps to measure similarities between features
    G = torch.mm(features, features.t())

    # 계산된 그램 행렬을 특성 맵의 전체 차원으로 나누어 정규화 -> 행렬 크기에 따른 스케일 조정하여 일반화 역할
    return G.div(a * b * c * d)


# Define a class for Content Loss
class ContentLoss(nn.Module):
    # Initalize the class
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    # Define forward
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# Define a class for Style Loss
class StyleLoss(nn.Module):
    # Initalize the class
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    # Define foward
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)

        return input


# Calculate Gram Matrix