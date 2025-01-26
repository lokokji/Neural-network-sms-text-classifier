# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Step 2: Load Dataset
url = 'https://your-dataset-url'  # Replace with actual URL or path to dataset
data = pd.read_csv(url)

# Step 3: Preprocess the data
# Clean the text data (convert to lowercase, remove punctuation, etc.)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Remove punctuation
    return text

data['text'] = data['text'].apply(preprocess_text)

# Step 4: Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text']).toarray()

# Step 5: Convert labels into numerical values
y = data['label'].map({'ham': 0, 'spam': 1}).values  # "ham" -> 0, "spam" -> 1

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build a Neural Network Model
model = Sequential([
    layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer (sigmoid for binary classification)
])

# Step 8: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 9: Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Step 10: Define the predict_message function
def predict_message(message):
    # Preprocess the input message
    message = preprocess_text(message)
    message_vector = vectorizer.transform([message]).toarray()
    
    # Predict the probability of being "spam" (1) or "ham" (0)
    prediction = model.predict(message_vector)
    
    # Return the probability and the label (ham or spam)
    probability = prediction[0][0]
    label = 'spam' if probability >= 0.5 else 'ham'
    return [probability, label]

# Step 11: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Step 12: Test the predict_message function
test_message = "Congratulations! You've won a $1000 gift card. Claim now!"
prediction = predict_message(test_message)
print(f"Prediction for message: {test_message}")
print(f"Probability: {prediction[0]} - Label: {prediction[1]}")
