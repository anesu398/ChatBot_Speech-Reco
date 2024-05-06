import os
import speech_recognition as sr
import pyttsx3
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Initialize speech recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Directory and database for storing chatbot data
DATA_DIR = "chatbot_data"
DATABASE = "chatbot.db"

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
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input TEXT,
        output TEXT
    )
    """)
    conn.commit()
    conn.close()

def store_data(input_text, output_text):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO interactions (input, output) VALUES (?, ?)", (input_text, output_text))
    conn.commit()
    conn.close()

def train_classifier():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT input, output FROM interactions")
    data = cursor.fetchall()
    conn.close()

    inputs = [row[0] for row in data]
    outputs = [row[1] for row in data]

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
chatbot()
