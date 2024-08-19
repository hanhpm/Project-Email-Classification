import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from Predict import preprocess_text
from Feature_extraction import create_dictionary, create_features

def train_model():
    DATASET_PATH = 'data/2cls_spam_text_cls.csv'
    df = pd.read_csv(DATASET_PATH)

    messages = df['Message'].values.tolist()
    labels = df['Category'].values.tolist()

    # Preprocess the text
    messages = [preprocess_text(message) for message in messages]

    # Create dictionary and feature vectors
    dictionary = create_dictionary(messages)
    X = np.array([create_features(tokens, dictionary) for tokens in messages])

    # Encode the labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Split the dataset
    VAL_SIZE = 0.2
    TEST_SIZE = 0.125
    SEED = 0

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=TEST_SIZE, shuffle=True, random_state=SEED)

    # Train the model
    model = GaussianNB()
    print("Start to train: ")
    model.fit(X_train, y_train)
    print("Train successfully done!")

    # Evaluate the model
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f'Val accuracy: {val_accuracy}')
    print(f'Test accuracy: {test_accuracy}')

    return model, dictionary, le

if __name__ == "__main__":
    train_model()
