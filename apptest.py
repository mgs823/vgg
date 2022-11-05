import numpy
from PIL import Image
from clsflower import predict

file_up="roses.jpg"
labels = predict(file_up)

for i in labels:
    print("预测 (名字)", i[0], ",   得分: ", i[1].astype(float))

    print(type(i[1]))