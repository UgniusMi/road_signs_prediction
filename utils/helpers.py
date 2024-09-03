import tkinter as tk
import os
from tkinter import filedialog
from termcolor import colored

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