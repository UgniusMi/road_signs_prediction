from database import *
from graphs import show_classes_graph
from model_cnn import start_training_model

def main_menu():
    create_db()
    print(f"SQL database created to this directory {databasepath}")
    create_one_csv()
    print(f"Train data csv file created to this directory {output_csv}")
    create_test_csv()
    print(f"Test data csv file created to this directory {test_images_path}")


    while True:
        print("\nWelcome to the Ugnius Road Signs Recognition Project!")
        print("1. Manage data ")
        print("2. Show Image classes graph")
        print("3. Train CNN model")
        print("4. Exit")

        choice = input("\nPlease enter your choice: ")
        
        if choice == '1':
            while True:
                print("1. Add your photo to train_data.db ")
                print("2. Add your photo to test_data.db")
                print("3. Generate default train db records")
                print("4. Generate default test db records")
                print("5. Back to main menu")
                choice = input("\nPlease enter your choice: ")
                if choice == '1':
                    path = input("Write your image path: ")
                    classid = int(input("Enter class id: "))
                    add_trainData(path, classid)
                elif choice == '2':
                    path = input("Write your image path: ")
                    classid = int(input("Enter class id: "))
                    add_testData(path, classid)
                elif choice == '3':
                    table_name = 'train_data'
                    csv_file_path = 'datasets/train_data.csv'
                    import_csv_data_to_table(table_name,csv_file_path)
                elif choice == '4':
                    table_name = 'test_data'
                    csv_file_path = 'datasets/test_data.csv'
                    import_csv_data_to_table(table_name,csv_file_path)
                elif choice == '5':
                    print("You have returned to the main menu")
                    break  

        elif choice == '2':
            train_df, test_df = get_classes_from_test_and_train_data()
            show_classes_graph(train_df, test_df)
        elif choice == '3':
            print("Start Training CNN model")
            start_training_model()
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break  
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
