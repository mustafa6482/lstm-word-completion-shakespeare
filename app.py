from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import tensorflow as tf  # Ensure TensorFlow is imported
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model and tokenizer
lstm_model = tf.keras.models.load_model('lstm_model.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

sequence_length = 5  # Match this with your training sequence length

def predict_next_words(model, tokenizer, seed_text, num_words=5):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length)
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = tokenizer.index_word[predicted[0]]
        seed_text += " " + output_word
    return seed_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    seed_text = request.form['text']
    predicted_text = predict_next_words(lstm_model, tokenizer, seed_text)
    return jsonify({'predicted': predicted_text})

if __name__ == '__main__':
    app.run(debug=True)
