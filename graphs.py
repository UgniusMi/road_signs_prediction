import matplotlib.pyplot as plt

def show_classes_graph(train_df, test_df):
    train_classes = train_df['ClassId'].value_counts()
    test_classes = test_df['ClassId'].value_counts() 
    train_classes = train_classes.sort_index(ascending=True)
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    train_classes.plot(kind='bar')
    plt.title('Train class Frequency ')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    test_classes = test_classes.sort_index(ascending=True)
    plt.subplot(1, 2, 2)
    test_classes.plot(kind='bar')
    plt.title('Test class Frequency ')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

