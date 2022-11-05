import streamlit as st
from PIL import Image
from clsflower import predict
import time
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("麻广森 简单分类网站")
st.write("")
st.write("")
# option = st.selectbox(
#      '选择要使用的模型?',
#      ('resnet50', 'resnet101', 'densenet121','shufflenet_v2_x0_5','mobilenet_v2'))
""
option2 = st.selectbox(
     '选择一个图片',
     ('daisy', 'roses'))

file_up = st.file_uploader("上传一张图像", type="jpg")
if file_up is None:
    if option2 =="daisy":
        image=Image.open("daisy.jpg")
        file_up="daisy.jpg"
    else:
        image=Image.open("roses.jpg")
        file_up="roses.jpg"
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("稍等...")
    # 最关键的一步，调用模型进行预测
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    st.success('成功识别')
    # st.write("名字", pre[0], ",   得分: ",pre[1])
    for i in labels:
        a=float(i[1])
        st.write("名字", i[0], ",   得分: ", round(a,4))

    # print(t2-t1)
    # st.write(float(t2-t1))
    # st.write("")
    # st.metric("", "FPS:   " + str(fps))

else:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("稍等...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    st.success('成功识别')
    # st.write("名字", pre[0], ",   得分: ", pre[1])
    for i in labels:
        a = float(i[1])
        st.write("名字", i[0], ",   得分: ", round(a, 4))

    # print(t2-t1)
    # st.write(float(t2-t1))
    st.write("")

