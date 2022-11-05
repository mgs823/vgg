import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
# torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
# torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
# torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
import torch.optim as optim
from tqdm import tqdm

from model import vgg


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # 定义一个把数据集的照片处理得到向量的操作
    data_transform = {  # transforms.Compose这个类的主要作用是串联多个图片变换的操作。
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     # transforms.RandomResizedCrop(224)将给定图像随机裁剪为不同的大小和宽高比，
                                     # 然后缩放所裁剪得到的图像为制定的大小；（即先随机采集，然后对裁剪得到的图像缩放为同一大小），默认scale=(0.08, 1.0)

                                     transforms.RandomHorizontalFlip(),
                                     # transforms.RandomHorizontalFlip  随机水平翻转
                                     transforms.ToTensor(),  # transforms.ToTensor() 将给定图像转为Tensor
                                     # transforms.Normalize(） 归一化处理
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 把数据集的照片处理得到向量
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # datasets.ImageFolder
    # root：图片存储的根目录，即各类别文件夹所在目录的上一级目录，在下面的例子中是’./ data / train /’。
    # transform：对图片进行预处理的操作（函数），原始图片作为输入，返回一个转换后的图片。
    # target_transform：对图片类别进行预处理的操作，输入为
    # target，输出对其的转换。如果不传该参数，即对target不做任何转换，返回的顺序索引0, 1, 2…
    # loader：表示数据集加载方式，通常默认加载方式即可。
    # 生成的dataset 有以下成员变量:
    # self.classes：用一个list保存类别名称
    # self.class_to_idx：类别对应的索引，与不做任何转换返回的 对应
    # self.imgs：保存(img - path,class ) tuple的 list，
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())  # 保存花种类名称
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)  # 花种类名称写入文件
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8  # 每批次训练8个
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # num_workers 用多少个子进程加载数据
    print('Using {} dataloader workers every process'.format(nw))

    # torch.utils.data.DataLoader
    # 该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，
    # 后续只需要再包装成Variable即可作为模型的输入，因此该接口有点承上启下的作用，比较重要
    #
    # 参数：

    # dataset(Dataset) – 加载数据的数据集。
    # batch_size(int, optional) – 每个batch加载多少个样本(默认: 1)。
    # shuffle(bool, optional) – 设置为True时会在每个epoch重新打乱数据(默认: False).
    # sampler(Sampler, optional) – 定义从数据集中提取样本的策略，即生成index的方式，可以顺序也可以乱序
    # num_workers(int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    # 处理验证集
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    # 开始确定网络模型
    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=5, init_weights=True)
    # vgg(model_name="vgg16", **kwargs):model = VGG(make_features(model_name), **kwargs)
    #
    # VGG(self, features, num_classes=1000, init_weights=False):
    net.to(device)
    # 这行代码的意思是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
    loss_function = nn.CrossEntropyLoss()
    # nn.CrossEntropyLoss是pytorch下的交叉熵损失，用于分类任务使用
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # 为了使用torch.optim，需先构造一个优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数。
    # 要构建一个优化器optimizer，你必须给它一个可进行迭代优化的包含了所有参数（所有的参数必须是变量s）的列表。 然后，您可以指定程序优化特定的选项，例如学习速率，权重衰减等。
    # 训练次数
    epochs = 30
    best_acc = 0.0  # 最好准确率
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)  # 训练集的个数
    for epoch in range(epochs):
        # train
        net.train()  # 开始训练
        # b) model.train() ：启用 BatchNormalization 和Dropout。
        # 在模型测试阶段使用model.train() 让model变成训练模式，此时dropout和batch
        # normalization的操作在训练q起到防止网络过拟合的问题。
        running_loss = 0.0  # 损失
        # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，
        # 用户只需要封装任意的迭代器 tqdm(iterator)。
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()  # 一些优化算法，如共轭梯度和LBFGS需要重新评估目标函数多次，必须清除梯度
            outputs = net(images.to(device))  # 输出是预测值
            loss = loss_function(outputs, labels.to(device))  # 损失，通过预测和标签计算
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # step()方法来对所有的参数进行更新

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()  # 开始验证，计算准确率
        # a) model.eval()，不启用
        # BatchNormalization和Dropout。此时pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
        # 不然的话，一旦test的batch_size过小，很容易就会因BN层导致模型performance损失较大；
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data  # 获取验证数据
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]  # 预测值的种类的索引
                # output = torch.max(input, dim)
                # 输入
                # input是softmax函数输出的一个tensor
                # dim是max函数索引的维度0 / 1，0是每列的最大值，1是每行的最大值
                # 输出
                # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。

                # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False。
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num  # 预测准确个数/总数
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:  # 保留最好的准确率（模型）
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
