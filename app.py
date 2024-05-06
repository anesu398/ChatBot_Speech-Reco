import os
import speech_recognition as sr
import pyttsx3
import pymongo
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize speech recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['chatbot_db']
interactions_collection = db['interactions']

def recognize_speech():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print("Error fetching results; {0}".format(e))
        return ""

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def initialize_database():
    interactions_collection.create_index([('input', pymongo.TEXT)])

def store_data(input_text, output_text):
    interactions_collection.insert_one({'input': input_text, 'output': output_text})

def train_classifier():
    inputs = []
    outputs = []
    for interaction in interactions_collection.find():
        inputs.append(interaction['input'])
        outputs.append(interaction['output'])

    # Preprocess inputs and outputs
    preprocessed_inputs = [preprocess_text(input_text) for input_text in inputs]
    preprocessed_outputs = [preprocess_text(output_text) for output_text in outputs]

    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(preprocessed_inputs, preprocessed_outputs)
    return model

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    # Join tokens back into a string
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

def chatbot():
    initialize_database()
    model = train_classifier()
    speak_text("Hello, how can I help you?")
    while True:
        text = recognize_speech()
        intent = model.predict([text])[0]
        if intent == "greet":
            response = "Hi there!"
        elif intent == "goodbye":
            response = "Goodbye!"
            speak_text(response)
            break
        else:
            response = "I'm sorry, I didn't understand that."
        speak_text(response)

        store_data(text, response)

# Example usage:
if __name__ == "__main__":
    chatbot()
