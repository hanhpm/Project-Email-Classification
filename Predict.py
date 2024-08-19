import numpy as np
from Preprocessing import preprocess_text
from Feature_extraction import create_features

def predict(text, model, dictionary, label_encoder):
    processed_text = preprocess_text(text)
    features = create_features(processed_text, dictionary)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    prediction_cls = label_encoder.inverse_transform(prediction)[0]
    return prediction_cls

if __name__ == "__main__":
    from Model_training import train_model

    model, dictionary, le = train_model()
    test_input = 'I am actually thinking a way of doing something useful'
    prediction_cls = predict(test_input, model, dictionary, le)
    print(f'Prediction: {prediction_cls}')
