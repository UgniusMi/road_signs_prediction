import matplotlib.pyplot as plt
import seaborn as sns

def show_classes_graph(train_df, test_df):
    train_classes = train_df['ClassId'].value_counts()
    test_classes = test_df['ClassId'].value_counts() 
    
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    train_classes.plot(kind='bar')
    plt.title('Train class Frequency ')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

   
    plt.subplot(1, 2, 2)
    test_classes.plot(kind='bar')
    plt.title('Test class Frequency ')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_F1_score_by_class(report_df):
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=report_df.index[:-3], y=report_df['f1-score'][:-3])
    plt.title('F1 Score per Class')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)  # F1 scores are between 0 and 1
    plt.show()
