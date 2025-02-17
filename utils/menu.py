from database.database import *
from utils.predict_class import *
from utils.graphs import show_classes_graph
from models.model_cnn import *
from models.model_rf import start_training_rf
import random
from termcolor import colored
from utils.helpers import *
import pyfiglet


big_text = pyfiglet.figlet_format("Ugnius  Road  Signs")
big_text += pyfiglet.figlet_format("Recognition  Project!")
print(big_text)

def main_menu():
    create_db()
    print(f"SQL database created to this directory {databasepath}")
    create_one_csv()
    print(f"Train data csv file created to this directory {output_csv}")
    create_test_csv()
    print(f"Test data csv file created to this directory {test_csv_path}")

    random.seed(10)
    np.random.seed(10)
    tf.random.set_seed(10)
    
    while True:
        print(colored("\nWelcome to the Main menu\n", "green"))
        print("1. Manage data ")
        print("2. Train model menu")
        print("3. Prediction menu")
        print("4. Open project overview document ")
        print(colored("\n5. Exit", "red"))
        

        choice = input("\nPlease enter your choice: ")
        if choice == '1':

            while True:
                print(colored("\nManage data menu:\n", "light_magenta"))
                print("1. Add your photo to train_data.db ")
                print("2. Add your photo to test_data.db")
                print("3. Generate default train db records")
                print("4. Generate default test db records")
                print(colored("\n5. Back to main menu", "light_cyan"))

                choice = input("\nPlease enter your choice: ")
                if choice == '1':
                    path = browse_file()
                    if path:
                        for classid, description in classes.items():
                            print(f"{classid}: {description}")
                        classid = get_valid_classid()
                        add_trainData(path, classid)
                        print(colored(f"Your image with ({path}) is successfully added to train_data table", "light_green"))

                elif choice == '2':
                    path = browse_file()
                    if path:
                        for classid, description in classes.items():
                            print(f"{classid}: {description}")
                        classid = get_valid_classid()
                        add_testData(path, classid)
                        print(colored(f"Your image with path ({path}) is successfully added to test_data table", "light_green"))

                elif choice == '3':
                    table_name = 'train_data'
                    csv_file_path = output_csv
                    import_csv_data_to_table_with_cleanup(table_name,csv_file_path)

                elif choice == '4':
                    table_name = 'test_data'
                    csv_file_path = test_csv_path
                    import_csv_data_to_table_with_cleanup(table_name,csv_file_path)

                elif choice == '5':
                    print("You have returned to the main menu")
                    break  
                else:
                 print(colored("Invalid choice. Please try again.\n", "red"))

        elif choice == '2':
            while True:
                print(colored("\nWelcome to Model training menu:\n", "light_magenta"))
                print("1. Show Image classes graph")
                print("2. Train CNN model")
                print("3. Train Random forest model")
                print(colored("\n4. Back to main menu", "light_cyan"))

                choice = input("\nPlease enter your choice: ")
                if choice == '1':
                    train_df, test_df = get_classes_from_test_and_train_data()
                    show_classes_graph(train_df, test_df)
                elif choice == '2':
                    start_training_cnn(learning_rate=LR, dropout_rate=DROP_RT, conv_filters=CONV_FILTER,dense_layer=DENSE_L, epochs=EPOCHS, batch_size=BATCH)
                elif choice == '3':
                    start_training_rf() 
                elif choice == '4':
                    print("You have returned to the main menu")
                    break  
                else:
                 print(colored("Invalid choice. Please try again.\n", "red"))

        elif choice == '3':
        
            while True:
                print(colored("\nWelcome to prediction menu:\n", "light_magenta"))

                model_path = 'savedmodels\\cnn982.keras' 
                model_path_rf ='savedmodels\\rfTEST.pkl'

                print("1. Predict your image by path with CNN model")
                print("2. Predict your image by path with Randomforest model")
                print("3. Predict all test data with CNN model")
                print("4. Show saved CNN model metrics")
                print(colored("\n5. Back to main menu", "light_cyan"))

                choice = input("\nPlease enter your choice: ")
                if choice == '1':
                    image_path = browse_file()  
                    if image_path:  
                        cnn_prediction(image_path, model_path, classes)
                        
                elif choice == '2':
                    image_path = browse_file()
                    if image_path:
                        rf_prediction(image_path,model_path_rf,classes)

                elif choice == '3':
                    predict_all_test(model_path)
                    print(colored("All test images predicted and saved in Predicted column", "light_green"))

                elif choice == '4':
                    show_cnn_model_metrics(model_path)

                elif choice == '5':
                    print("You have returned to the main menu")
                    break  
                else:
                 print(colored("Invalid choice. Please try again.\n", "red"))

        elif choice == '4':
            doc_path = "datasets\project_overview.rtf"
            open_word_doc(doc_path)

        elif choice == '5':
            print(colored("Exiting the program. Goodbye!", "dark_grey"))
            break  
        else:
            print(colored("Invalid choice. Please try again.\n", "red"))