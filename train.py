import os
import cv2..............................................zW
import numpy as np
import pickle
from PIL import Image
from sklearn.linear_model import LogisticRegression
from keras_facenet import FaceNet
from mtcnn import MTCNN


def extractFace(image: Image):

    faces = detector.detect_faces(np.array(image))

    faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)

    try:
        x, y, w, h = faces[0]['box']
    except:
        return image

    return image.crop((x, y, x + w, y + h))


def get_dataset(dtype="train", crop_face=False):
    images = []
    targets = []
    with open(os.path.join(dataset, dtype + '.csv'), 'r') as file:
        for line in file.readlines():
            path, cls = map(lambda x: x.strip(), line.split(','))

            img = extractFace(Image.open(
                path)) if crop_face else Image.open(path)

            images.append(cv2.resize(np.array(img), (160, 160)))

            targets.append(int(cls))

    # Convert to numpy
    X = np.array(images)
    y = np.array(targets)

    return X, y


def get_label():
    labels = {}

    with open(os.path.join(dataset, 'label.txt')) as file:
        for i, person in enumerate(file.readlines()):
            labels[int(i)] = person.strip()

    return labels


if __name__ == "__main__":
    detector = MTCNN()
    embedder = FaceNet()
    model = LogisticRegression()

    dataset = "./data/fsktm"

    X_train, y_train = get_dataset(dtype="train")
    X_test, y_test = get_dataset(dtype="test")
    labels = get_label()

    print('X_Train data shape=', X_train.shape)
    print('X_Test data shape=', X_test.shape)
    print('y_Train data shape=', y_train.shape)
    print('y_Test data shape=', y_test.shape)

    # Use the pretrained facenet for face embedding
    X_train = embedder.embeddings(X_train)
    X_test = embedder.embeddings(X_test)
    print('Train embed shape=', X_train.shape)
    print('Test embed shape=', X_test.shape)

    model.fit(X_train, y_train)
    print(f"Precision: {model.score(X_test, y_test)}")

    pickle.dump(model, open("./models/fsktm.pkl", 'wb'))
