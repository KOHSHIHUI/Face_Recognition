import streamlit as st
from keras_facenet import FaceNet
from mtcnn import MTCNN
from PIL import Image, ImageDraw, ImageFont
import pickle
import os
import cv2
import numpy as np
import pandas as pd


def extractFaces(image: Image, faces):

    output = []

    for face in faces:
        x, y, w, h = face['box']
        output.append(image.crop((x, y, x + w, y + h)))

    return output


def drawBoundingBoxesAndLabel(image: Image, faces, preds):
    tmp = image.copy()
    draw = ImageDraw.Draw(tmp)
    for i, face in enumerate(faces):
        x, y, w, h = face['box']
        label = preds[i].split('-')[-1].strip()
        font = ImageFont.truetype("arial.ttf", 50)

        text_bbox = draw.textbbox((x, y - 50), label, font=font)

        draw.rectangle((x, y, x + w, y + h), outline="red", width=10)

        draw.rectangle(text_bbox, width=2, fill="black")
        draw.text((x, y - 50), label, fill="white", font=font)

    return tmp


def getDataframe(preds, confs):
    names = []
    matric_no = []
    found = []

    for i, record in labels.items():
        names.append(record.split('-')[0].strip())
        matric_no.append(record.split('-')[1].strip())
        found.append("Present" if record in preds else "Absent")

    df = pd.DataFrame({
        "Names": names,
        "Matric No": matric_no,
        "Present": found,
    })

    return df


def get_label(dataset):
    labels = {}

    with open(os.path.join(dataset, 'label.txt')) as file:
        for i, person in enumerate(file.readlines()):
            labels[int(i)] = person.strip()

    return labels


def predict(image: Image, with_label=True):
    x = np.array(image)
    x = cv2.resize(x, (160, 160))
    x = x.reshape(1, 160, 160, 3)
    x = embedder.embeddings(x)
    conf = max(model.predict_proba(x)[0])
    if (conf < 0.15):  # 0.25 is the threshold
        print("Not recognizable !")
        return "", -1
    else:
        y_s = model.predict(x)[0]
        return labels[y_s] if with_label else y_s, conf


@st.cache(allow_output_mutation=True)
def loadModels():
    # Load models
    detector = MTCNN()
    embedder = FaceNet()
    model = pickle.load(open('./models/fsktm.pkl', 'rb'))
    labels = get_label("./data/fsktm")

    return embedder, detector, model, labels


def faceDetection(my_upload):
    image = Image.open(my_upload)
    preds = []
    confs = []

    faces = detector.detect_faces(np.array(image))

    cropped_faces = extractFaces(image, faces)

    for face in cropped_faces:
        output = predict(face)

        preds.append(output[0])
        confs.append(output[1])

    print(preds)

    output = drawBoundingBoxesAndLabel(image, faces, preds)

    return image, output, preds, confs

def color_present(val):
    color = '#00FF0C' if val == "Present" else '#FC1717'
    return f'background-color: {color}'

st.set_page_config(layout="wide", page_title="FSKTM Attendance System")

st.title("Automated Attendance Record System")

with st.spinner(text="Loading models ..."):
    embedder, detector, model, labels = loadModels()

st.sidebar.write("## Upload or Select :gear:")

my_upload = st.sidebar.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"],
)

col1, col2 = st.columns([7, 3])

if my_upload is not None:
    with st.spinner(text="Predicting ..."):
        image, output, preds, confs = faceDetection(my_upload)
        df = getDataframe(preds, confs)
        df = df.style.applymap(color_present, subset=['Present'])
        df.to_excel("./results/result.xlsx")

    col1.image(output)

    with open("./results/result.xlsx", "rb") as file:
        col2.download_button("Download Result", data=file,
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             file_name='result.xlsx'
                             )

    col2.table(df)
