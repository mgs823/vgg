import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        #  net = vgg(model_name=model_name, num_classes=5, init_weights=True)
        super(VGG, self).__init__()
        self.features = features  # 由make_features函数确定，处理特征值的方法
        #  features确定卷积和池化的操作顺序，.classifier 确定全连接分类的参数
        self.classifier = nn.Sequential(
            # 卷积后的结果是512x7x7,输入512*7*7xbatch_size,输出4096xbatch_size
            # 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
            # 同时以神经网络模块为元素的有序字典也可以作为传入参数。
            nn.Linear(512 * 7 * 7, 4096),  # 全连接层fc，矩阵展平
            nn.ReLU(True),
            nn.Dropout(p=0.5),  # 防止过拟合
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    # 前向传播
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)  # 卷积提取特征
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)  # 展平
        # N x 512*7*7
        x = self.classifier(x)  # 全连接分类
        return x

    def _initialize_weights(self):  # 初始化各层权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 判断是否相同
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)  # 初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 确定每一层是卷积层还是池化，及其参数（卷积核大小，步长等）
def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:  # 遍历层表
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {  # M代表池化，数字代表卷积出的个数
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 确定运行的是哪一个版本的vgg
def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]  # 确定运行的是哪一个版本的vgg

    model = VGG(make_features(cfg), **kwargs)
    # cfg经过make_features后，已经确定每一层的分布
    return model
