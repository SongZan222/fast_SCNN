import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision import datasets
import torchvision.transforms as T
from timeit import default_timer as timer
import time
from torchvision.models.mobilenetv2 import Conv2dNormActivation


# 0.0 Hyper-parameters 超参数
LOCAL_TIME = time.localtime(time.time())
TIME_STRING = time.strftime("%Y_%m_%d_%H_%M", LOCAL_TIME)
DATASETS_PATH = r'E:\Dataset'
MODELS_STORAGE_PATH = r'D:\python_project\pythonProject\models'
DATASET_NAME = 'MNIST'    # Dataset name in TUDataset TUDataset中的数据集名称
HEIGHT = 128
WIDTH = 128
TRANS_TYPE = '01'
NET_NAME = 'CNN'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # Use GPU first 优先使用GPU
LEARNING_RATE = 0.01   # Learning rate 学习率
HIDDEN_DIMENSION = 16  # Hidden layer dimension 隐藏层维度
EPOCH = 2            # epoch Recommended value greater than 500 建议大于500
TEST_TIMES = 10       # Number of tests  测试次数 Suggestion 100
BATCH_SIZE_TRAIN = 64 # batch size of the training set 训练集batch大小
BATCH_SIZE_TEST = 500  # batch size of the testing set 测试集batch大小
RATIO = 0.75           # Discard rates in pooled models 池化模型中的丢弃率
DROP_OUT = 0.05
DYNAMIC_LEARNING_RATE = False   # Whether to adjust the learning rate with the epoch 是否随epoch调整学习率
SAVE_TRAINING_RESULTS = True # Whether to store  model是否存储训练结果
ACC = 0
DVA = 0

def plt_line_graph(list_x, list_y1):
    '''

    :param list_x: The horizontal axis of the line graph，折线图的横轴
    :param list_y1: The first set of data that needs to be drawn 第一组数据
    :param list_y2: The second set of data that needs to be drawn 第二组数据
    :return: void Draw the line graph
    '''
    plt.plot(
        list_x,
        list_y1,
        label="acc",
        color='#58006E'
    )
    # y1 Average value   绘制y1平均值
    print(list_x)
    print(list_y1)
    y1_average = [sum(list_y1) / (len(list_y1)), sum(list_y1) / (len(list_y1))]
    x_first_and_end = [list_x[0], list_x[-1]]
    print('the average acc is ')
    print(y1_average[0])
    plt.plot(
        x_first_and_end,
        y1_average,
        label="average acc:"+str(format(y1_average[0],'.4f')),
        color='#E5439A',
        linestyle='-.',
        markersize=15
    )
    # 绘制y1标准差
    np.std(list_y1, ddof=1)
    avge =sum(list_y1) / (len(list_y1))
    y1_std_down = [avge-np.std(list_y1, ddof=1), avge-np.std(list_y1, ddof=1)]
    y1_std_up = [avge+np.std(list_y1, ddof=1), avge+np.std(list_y1, ddof=1)]
    x_first_and_end = [list_x[0], list_x[-1]]
    print('the standard deviation is ')
    print(y1_std_down[0])
    plt.plot(
        x_first_and_end,
        y1_std_down,
        color='#FC9F81',
        label = 'standard deviation:±'+str(format(np.std(list_y1, ddof=1),'.4f')),
        linestyle='-.',
        markersize=15
    )
    plt.plot(
        x_first_and_end,
        y1_std_up,
        color='#FC9F81',
        linestyle='-.',
        markersize=15
    )

    # plt.plot(
    #     list_x,
    #     list_y2,
    #     label="Accuracy Rate after N epochs",
    #     color='#F6573D',
    # )
    # #  y2 Average value 绘制y2平均值
    # y2_average = [sum(list_y2) / len(list_y2), sum(list_y2) / len(list_y2)]
    # x_first_and_end = [list_x[0], list_x[-1]]
    # plt.plot(
    #     x_first_and_end,
    #     y2_average,
    #     label="average accuracy",
    #     color='#FC9F81',
    #     linestyle='-.',
    #     markersize=15
    # )
    plt.legend(loc='best')
    # Control Y-axis range and step size 控制y轴范围和步长
    # plt.yticks(y_ticks[10:90:10])
    plt.grid(  # Background dotted line grid 背景虚线网格
        True,
        linestyle='--',
        alpha=0.25
    )
    # The name of the axis and the name of the chart 轴的名称和表的名称
    plt.xlabel("idx", fontdict={'size': 16})
    plt.ylabel("loss and accuracy", fontdict={'size': 16})
    # plt.title(TEST_ITEM, fontdict={'size': 16})
    # print(get_test_item()+'_'+DATASET_NAME+'_'+format(EPOCH))
    print(str(format(y1_average[0],'.4f'))+'±'+str(format(np.std(list_y1, ddof=1),'.4f')))
    global ACC
    global DVA
    ACC = str(format(y1_average[0],'.3f'))
    DVA = str(format(np.std(list_y1, ddof=1),'.3f'))
    plt.show()

# Drawing matrix 绘制矩阵
def plt_matrix(m, name='the image of matrix'):
    '''
    :param m: ndarray dim = 2
    :param name: string
    :return: void and Draw the matrix
    '''
    m = plt.matshow(m, cmap=plt.cm.RdPu)
    plt.colorbar(m.colorbar, fraction=0.25)
    plt.title(name)
    plt.show()


class ConvBNReLU(nn.Module):
    '''
    https://blog.csdn.net/Mr_health/article/details/125978193
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class DSConv(nn.Module):
    """
    Depthwise Separable Convolutions
    https://blog.csdn.net/kangdi7547/article/details/117925389
    stride 步长
    groups 分组卷积
    """

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                dw_channels,
                dw_channels,
                3,
                stride,
                1,
                groups=dw_channels,
                bias=False
            ),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        # https://blog.csdn.net/m0_51004308/article/details/118000391
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        # https://blog.csdn.net/qq_50001789/article/details/120297401
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x

class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = Conv2dNormActivation(3, dw_channels1, 3, 2)
        self.dsconv1 = DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x

class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # CBR
            ConvBNReLU(in_channels, in_channels * t, 1),
            # DW
            DWConv(in_channels * t, in_channels * t, stride),
            # Conv + BN
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out

class DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x

