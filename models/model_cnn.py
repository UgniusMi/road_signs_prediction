import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import layers # type: ignore
from database.database import get_engine
from utils.graphs import plot_training_history


LR =0.0007
DROP_RT=0.6 
CONV_FILTER=[32, 64, 128] 
DENSE_L = 128
EPOCHS=15
BATCH=32

def preprocess_images(image_paths, img_size=(48, 48)): #(5,5)kartu su 48x48
    images = []
    for path in image_paths:
        img = cv2.imread(path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        img = cv2.resize(img, img_size)  
        img = img / 255.0  
        images.append(img)
    return np.array(images)

def load_data(train_table_name, test_table_name):
    engine = get_engine() 
    train_df = pd.read_sql_table(train_table_name, engine)
    test_df = pd.read_sql_table(test_table_name, engine)

    print(f"TRAIN DF: {train_df.head(5)}")
    print(f"TEST DF: {test_df.head(5)}")

    X_train = preprocess_images(train_df['ImagePath'].tolist())
    y_train = train_df['ClassId'].values
    X_test = preprocess_images(test_df['ImagePath'].tolist())
    y_test = test_df['ClassId'].values

    return X_train, X_test, y_train, y_test

def create_model(input_shape, num_classes, learning_rate=0.0006, dropout_rate=0.6, conv_filters=[32, 64, 128], dense_layer= 128):
    # data_augmentation = tf.keras.Sequential([
    # layers.RandomFlip("horizontal"),
    # layers.RandomContrast(0.3)
# ])
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape),
        # data_augmentation,
        layers.Conv2D(conv_filters[0], (5, 5), activation='relu'), #(5,5) kartu 48x48
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(conv_filters[1], (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(conv_filters[2], (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(dense_layer, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']
                 )
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=2, min_lr=0.00001)

    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=epochs, 
                        batch_size=batch_size,
                        callbacks=[early_stopping, reduce_lr]
                       )
    return history

def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.3f}")
    return test_accuracy

def save_model(model, filename='savedmodels\\cnnTEST.keras'):
    model.save(filename)


def start_training_cnn(learning_rate=0.0006, dropout_rate=0.6, conv_filters=[32, 64, 128],dense_layer= 128, epochs=15, batch_size=32):
    
    X_train, X_test, y_train, y_test = load_data('train_data', 'test_data')
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    num_classes = len(np.unique(y_train))
    print("Number of classes:", num_classes)

    model = create_model(input_shape=(48, 48, 3), num_classes=num_classes, 
                         learning_rate=learning_rate, 
                         dropout_rate=dropout_rate, 
                         conv_filters=conv_filters,
                         dense_layer= dense_layer
                         )
    
    print(f"Drop rate: {dropout_rate}")
    print(f"conv filter {conv_filters}")
    print(f"DL {dense_layer}")
    print(f"Batch size {batch_size}")
    
    history = train_model(model, X_train, y_train, X_val, y_val, 
                          epochs=epochs, 
                          batch_size=batch_size
                          )

    evaluate_model(model, X_test, y_test)

    while True:
        save_option = input("Do you want to save the model? (yes/no): ").lower()
        if save_option == 'yes':
            save_model(model)
            print("Model has been saved.")
            break
        elif save_option == 'no':
            print("Model was not saved.")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
  
    
    plot_training_history(history)