import os
import numpy as np
from load_and_preprocess import prepare_data
from model import cnn_model, randomforest_model
from train_and_evaluate import train_model, evaluate_model
import cv2
from tensorflow.keras.models import load_model
from zSignclasses import classes

def start_training(train_table_name, test_table_name, model_type='cnn', epochs=10, batch_size=32, test_size=0.2, random_state=42):
    # Paruoškite duomenis
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = prepare_data(train_table_name, test_table_name, test_size, random_state)
    
    # Sukurkite modelį
    if model_type == 'cnn':
        model = cnn_model(input_shape=(42, 42, 3), num_classes=num_classes)
    elif model_type == 'random_forest':
        model = randomforest_model()
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        print("Training completed")
        return model, X_test, y_test
    
    # Treniruokite modelį
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    print("Training completed")
    

    
    return model, X_test, y_test

def predict(model, image_path):
    # Nuskaityti ir apdoroti paveikslėlį
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (42, 42))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Numatyti klasę
    y_pred = model.predict(img)
    predicted_class = y_pred.argmax(axis=1)[0]
    
    return predicted_class

if __name__ == '__main__':
    # Pavyzdys, kaip iškviesti start_training
    model, X_test, y_test = start_training(train_table_name='train_data', test_table_name='test_data', model_type='cnn', epochs=15, batch_size=32)
    evaluate_model(model, X_test)
    # Pavyzdys, kaip iškviesti predict funkciją
    image_path = r'C:\Users\Fatalas\Baigiamasis\datasets\test_images\11745.ppm'
    predicted_class = predict(model, image_path)
    print(f"Predicted class: {predicted_class}")