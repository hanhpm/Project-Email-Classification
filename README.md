# Project-Email-Classification

This project focuses on classifying emails as either "spam" or "ham" (not spam) using Natural Language Processing (NLP) techniques and a Naive Bayes classifier.

## 1. Project Structure

The project is organized into the following Python scripts:

- **Main.py**: The main script that orchestrates the entire email classification process.
- **Data_loader.py**: Loads the email dataset for training and testing the model.
- **Preprocessing.py**: Contains functions for preprocessing the email text, including tokenization, stopword removal, and stemming.
- **Model_training.py**: Handles the training of the Naive Bayes classifier and provides functionality for model evaluation.

## 2. Setup and Installation

To run this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourUsername/Project-Email-Classification.git
   cd Project-Email-Classification
   ```

2. **Install libraries and data packages**:
- All necessary libraries are noted in each files .py.
- Place dataset with data path.

## 3. Run the project 
    ```bash
    python Main.py
    ```
The script will:

- Load the data using Data_loader.py.
- Preprocess the data using Preprocessing.py.
- Train the Naive Bayes model using Model_training.py.
- Evaluate the model's performance on validation and test datasets.

