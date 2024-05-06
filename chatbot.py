import pymongo
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import speech_recognition as sr
import pyttsx3

# Initialize speech recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

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
    client = MongoClient('mongodb://localhost:27017/')
    db = client['chatbot_db']
    interactions_collection = db['interactions']
    interactions_collection.create_index([('input', pymongo.TEXT)])

def store_data(input_text, output_text):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['chatbot_db']
    interactions_collection = db['interactions']
    interactions_collection.insert_one({'input': input_text, 'output': output_text})

def train_classifier():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['chatbot_db']
    interactions_collection = db['interactions']

    inputs = []
    outputs = []
    for interaction in interactions_collection.find():
        inputs.append(interaction['input'])
        outputs.append(interaction['output'])

    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(inputs, outputs)
    return model
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


# Example usage:
initialize_database()
chatbot()  # Uncomment this line if you have a chatbot function defined

# Note: You need to define the chatbot function or remove the chatbot() call if not needed.
