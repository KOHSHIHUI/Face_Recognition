import os
import cv2
import numpy as np
import pickle
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from keras_facenet import FaceNet
from mtcnn import MTCNN


def extractFaces(image: Image, faces):

    output = []

    for face in faces:
        x, y, w, h = face['box']
        output.append(image.crop((x, y, x + w, y + h)))

    return output


def drawBoundingBoxesAndLabel(image: Image, faces, labels):
    tmp = image.copy()
    draw = ImageDraw.Draw(tmp)
    for i, face in enumerate(faces):
        x, y, w, h = face['box']
        label = labels[i]

        text_bbox = draw.textbbox((x, y - 13), label)

        draw.rectangle((x, y, x + w, y + h), outline="red", width=2)

        draw.rectangle(text_bbox, width=2, fill="black")
        draw.text((x, y - 13), label, fill="white")

    return tmp


def get_label(dataset):
    labels = {}

    with open(os.path.join(dataset, 'label.txt')) as file:
        for i, person in enumerate(file.readlines()):
            labels[int(i)] = person.strip()

    return labels


def predict(image: Image, with_label=True):
    x = np.array(image)
    x = cv2.resize(x, (160, 160))
    x = np.expand_dims(x, axis=0)
    x = embedder.embeddings(x)
    if (max(model.predict_proba(x)[0]) < 0.25):  # 0.25 is the threshold
        print("Not recognizable !")
        return ""
    else:
        y_s = model.predict(x)[0]
        return labels[y_s] if with_label else y_s


if __name__ == "__main__":
    detector = MTCNN()
    embedder = FaceNet()
    model = pickle.load(open('./models/fsktm.pkl', 'rb'))
    labels = get_label("./data/fsktm")

    image = Image.open(
        "./images/sample_2.jpg"
    )

    faces = detector.detect_faces(np.array(image))

    cropped_faces = extractFaces(image, faces)

    labels = [predict(face) for face in cropped_faces]

    print(labels)

    output = drawBoundingBoxesAndLabel(image, faces, labels)

    output.show()

    output.save("./results/sample_2.jpg")
