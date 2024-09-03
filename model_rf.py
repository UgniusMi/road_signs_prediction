import pandas as pd
import numpy as np
import cv2
import pickle
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from database import get_engine
from graphs import plot_F1_score_by_class

def load_images_with_hog(image_paths, target_size=(64, 64)):
    images = []
    for path in image_paths:
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size)
            img = img / 255.0
            hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
            images.append(hog_features)
        except (FileNotFoundError, cv2.error):
            print(f"Error loading image: {path}")
            continue
    return np.array(images)


def load_data(train_table_name, test_table_name):
    engine = get_engine() 
    train_df = pd.read_sql_table(train_table_name, engine)
    test_df = pd.read_sql_table(test_table_name, engine)

    print(f"TRAIN DF: {train_df.head(5)}")
    print(f"TEST DF: {test_df.head(5)}")

    X_train = load_images_with_hog(train_df['ImagePath'].tolist())
    y_train = train_df['ClassId'].values
    X_test = load_images_with_hog(test_df['ImagePath'].tolist())
    y_test = test_df['ClassId'].values

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    return rf_model

def metrics(rf_model,X_test, y_test,X_val,y_val):
    test_y_pred = rf_model.predict(X_test)
    val_y_pred = rf_model.predict(X_val)
    test_accuracy = accuracy_score(y_test, test_y_pred)
    val_accuracy = accuracy_score(y_val, val_y_pred)

    test_f1 = f1_score(y_test, test_y_pred, average='weighted')

    cr_for_graph = classification_report(y_test, test_y_pred, output_dict= True)
    cr = classification_report(y_test, test_y_pred)

    report_df = pd.DataFrame(cr_for_graph).transpose()

    return test_accuracy,val_accuracy, test_f1, report_df, cr
    
def save_model(model, filename='savedmodels\\rfTEST.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def start_training():
    X_train, X_test, y_train, y_test = load_data('train_data', 'test_data')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = train_random_forest(X_train, y_train)
    
    test_accuracy,val_accuracy, test_f1, report_df, cr = metrics(model,X_test, y_test,X_val,y_val)
    print(f"Classification report:\n{cr}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    save_model(model)

    plot_F1_score_by_class(report_df)






