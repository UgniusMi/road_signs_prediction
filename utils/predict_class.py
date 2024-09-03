import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from database.database import TestData, sessionmaker, get_engine
from models.model_cnn import load_data
from skimage.feature import hog
from sklearn.metrics import classification_report,f1_score, accuracy_score
from utils.graphs import  plot_F1_score_by_class

def load_cnn_model(model_path):
    return tf.keras.models.load_model(model_path)

def load_rf_model(model_path_rf):
    try:
        with open(model_path_rf, 'rb') as file:
            loaded_model = pickle.load(file)
            return loaded_model
    except Exception as e:
        print(f"Klaida įkeliant modelį: {str(e)}")
        exit(1)

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (42, 42))  
    img = img / 255.0 
    img = np.expand_dims(img, axis=0)  
    return img

def preprocess_image_for_rf(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Nepavyko įkelti paveiksliuko iš {image_path}")
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                           cells_per_block=(2, 2), block_norm='L2-Hys')
        return hog_features.reshape(1, -1)
    except Exception as e:
        print(f"Klaida apdorojant paveikslėlį {image_path}: {str(e)}")
        return None

def predict_class(model, model_type, X_test):
    if model_type == "rf":
        y_pred_proba = model.predict_proba(X_test) # random forest
        predicted_class = np.argmax(y_pred_proba, axis=1)[0]

    elif model_type == "cnn": 
        y_pred = model.predict(X_test) # cnn
        predicted_class = np.argmax(y_pred, axis=1)[0]
    
    return int(predicted_class)

def display_prediction(image_path, predicted_class, class_descriptions):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f'Predicted Class: {predicted_class} - {class_descriptions.get(predicted_class, "Unknown class")}')
    plt.show()


def cnn_prediction(image_path, model_path, class_descriptions):

    processed_image = load_and_preprocess_image(image_path)

    model = load_cnn_model(model_path)

    predicted_class = predict_class(model, "cnn", processed_image)
    print(f"Predicted class with CNN model: {predicted_class}")
    display_prediction(image_path, predicted_class, class_descriptions)


def rf_prediction(image_path, model_path_rf, class_descriptions):

    processed_image = preprocess_image_for_rf(image_path)
    model = load_rf_model(model_path_rf)
    predicted_class = predict_class(model, "rf", processed_image)
    print(f"Predicted class with Random forest model: {predicted_class}")
    
    display_prediction(image_path, predicted_class, class_descriptions)


def predict_all_test(model_path):
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    model_path = 'savedmodels\\cnnTEST.keras' # gal maine padaryt viesiem pasiekiam
    model = load_cnn_model(model_path)

    test_data = session.query(TestData).all()

    for test_record in test_data:
        processed_image = load_and_preprocess_image(test_record.ImagePath)

        predicted_class = predict_class(model, "cnn", processed_image)
        print(f"predicted class :{predicted_class}") 
        test_record.Predicted = predicted_class
        session.add(test_record)

    session.commit()
    session.close()

def show_cnn_model_metrics(model_path):
    
        model = load_cnn_model(model_path)
        X_train, X_test, y_train, y_test = load_data('train_data', 'test_data')

        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        print(f"Classification Report:\n {class_report}")
        print(f"Test F1 Score: {f1:.3f}")
        print(f"Test Accuracy: {acc:.3f}")

        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        plot_F1_score_by_class(report_df)


