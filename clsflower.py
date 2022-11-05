import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import vgg


def predict(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image

    img_path = image_path

    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)  # 处理测试图片为张量
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    # unsqueeze()函数起升维的作用,参数表示在哪个地方加一个维度。
    # 在第一个维度(中括号)的每个元素加中括号 0表示在张 量最外层加一个中括号变成第一维。
    # 图像的向量为一个列表，大小是224*224，unsqueeze后变为1 x 224*224的张量

    # read class_indict
    json_path = './class_indices.json'  # 读取花种类名称，字典格式
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = vgg(model_name="vgg16", num_classes=5).to(device)
    # load model weights
    weights_path = "./vgg16Net.pth"  # 加载训练好的权重
    # assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # 把权重填入模型，
    # map_location=torch.device('cpu')，意思是映射到cpu上，在cpu上加载模型，无论你这个模型从哪里训练保存的。
    # 一句话：map_location适用于修改模型能在gpu上运行还是cpu上运行
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()  # 开始验证,不启用 BatchNormalization 和 Dropout。
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        # queeze()
        # 函数的功能是维度压缩。返回一个tensor（张量），其中input中大小为1的所有维都已删除。
        #
        # 举个例子：如果 input的形状为(A×1×B×C×1×D)，那么返回的tensor的形状则为(A×B×C×D)
        predict = torch.softmax(output, dim=0)
        # torch.nn.Softmax中只要一个参数：来制定归一化维度如果是dim=0指代的是行，dim=1指代的是列。
        predict_cla = torch.argmax(predict).numpy()  # 返回最大值的索引，也就是预测种类的名称
        # torch.argmax(input, dim=None, keepdim=False)
        # 返回指定维度最大值的序号
        # dim给定的定义是：the demention to reduce.也就是把dim这个维度的，变成这个维度的最大值的index。

    # 打印种类名字和正确概率
    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # plt.title(print_res)

    # 分别打印每个种类的可能概率
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    pre=[class_indict[str(predict_cla)],predict[predict_cla].numpy()]
    return [(class_indict[str(i)], predict[i].numpy()) for i in range(len(predict))]
