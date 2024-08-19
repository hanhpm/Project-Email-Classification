from Model_training import train_model
from Predict import predict

if __name__ == "__main__":
    # Step 1: Train the model
    model, dictionary, le = train_model()

    # Step 2: Test prediction
    test_input = 'I am actually thinking a way of doing something useful'
    prediction_cls = predict(test_input, model, dictionary, le)
    print(f'Prediction: {prediction_cls}')
