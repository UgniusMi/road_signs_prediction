from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import csv
import os
from termcolor import colored
import tkinter as tk
from tkinter import filedialog

Base = declarative_base()

class TrainData(Base):
    __tablename__ = 'train_data'
    id = Column(Integer, primary_key=True)
    ImagePath = Column(String)
    ClassId = Column(Integer)
    
class TestData(Base):
    __tablename__ = 'test_data'
    id = Column(Integer, primary_key=True)
    ImagePath = Column(String)
    ClassId = Column(Integer)
    Predicted = Column(Integer) #su situ veliau


databasepath = "sqlite:///database/train_test.db"
rootpath= "datasets\\train_images"
output_csv='datasets\\train_data.csv'
test_images_path = 'datasets/test_images'

def get_engine():
    return create_engine(databasepath)

def create_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def create_one_csv():
    data = []
    for c in range(0, 43):
        prefix = os.path.join(rootpath, format(c, '05d')) + '/' 
        gt_file = os.path.join(prefix, 'GT-' + format(c, '05d') + '.csv')  
        
        with open(gt_file, 'r') as f:
            gt_reader = csv.reader(f, delimiter=';')
            next(gt_reader)  
            
            for row in gt_reader:
                img_path = os.path.join(prefix, row[0])  
                label = row[7]  
                data.append([img_path, label])
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ImagePath', 'ClassId']) 
        writer.writerows(data)
    
    print(f"Data saved to {output_csv}")

def create_test_csv():
    test_df = pd.read_csv('datasets/GT-final_test.csv', delimiter=';')
    test_df = test_df[['Filename', 'ClassId']]
    test_df['ImagePath'] = test_df['Filename'].apply(lambda x: os.path.join(test_images_path, x))
    test_df = test_df[['ImagePath','ClassId']]
    test_df.to_csv('datasets\\test_data.csv', index=False)
    return test_df

def import_csv_data_to_table_with_cleanup(table_name, csv_file_path):
    engine = get_engine()
    session = create_db()

    table_class = Base.metadata.tables.get(table_name)
    
    if table_class is not None:
        session.query(table_class).delete()
        session.commit()
        print(f"All records from {table_name} have been deleted.")
    else:
        print(f"Table {table_name} does not exist.")
        return
    
    all_rows = pd.read_csv(csv_file_path) 
    all_rows.to_sql(table_name, con=engine, if_exists='append', index=False)
    print(f"Data from {csv_file_path} has been imported into {table_name}.")


def add_trainData(image_path, classid):
    session = create_db()
    trainData = TrainData(ImagePath=image_path, ClassId=classid)
    session.add(trainData)
    session.commit()

def add_testData(image_path, classid):
    session = create_db()
    testData = TestData(ImagePath=image_path, ClassId=classid)
    session.add(testData)
    session.commit()

def get_classes_from_test_and_train_data():
    engine = get_engine() 
    train_df = pd.read_sql_table('train_data', engine)
    test_df = pd.read_sql_table('test_data', engine)
    return train_df, test_df

def get_valid_image_path():
    while True:
        image_path = input("\033[93mSpecify your image path \033[0m(example: C:\\Users\\Fatalas\\Desktop\\uploads\\yourimage.ppm): ")
        if os.path.isfile(image_path) and image_path.lower().endswith(('.ppm', '.jpg', '.jpeg', '.png')):
            return image_path
        else:
            print(colored("Invalid image path or format.", "red"))

def get_valid_classid(min_value=0, max_value=42):
    while True:
        try:
            classid = int(input(f"Enter class id (between {min_value} and {max_value}): "))
            if min_value <= classid <= max_value:
                return classid
            else:
                print(f"Invalid class id.")
        except ValueError:
            print(colored("Invalid input. Please enter a valid integer.", "red"))

def browse_file():
    print("Opening file dialog...")
    root = tk.Tk()
    root.withdraw()  # paslepia pagrindini tkinter langa
    root.attributes('-topmost', True)  # palaiko kad dialogo langas butu virsuje
    file_path = filedialog.askopenfilename(
        parent=root,
        title="Select a file",
        filetypes=(("Image files", "*.ppm *.jpg *.jpeg *.png"), ("All files", "*.*"))
    )
    root.destroy() 
    if file_path:
        print(f"Selected file: {file_path}")
    else:
        print("No file selected.")
    return file_path


