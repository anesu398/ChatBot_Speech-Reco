# Chatbot with MongoDB Integration By Anesu Ndava

This project implements a sophisticated chatbot with MongoDB integration for storing user interactions.

## Features

- **Speech recognition:** The chatbot can recognize speech input using the `SpeechRecognition` library.
- **Text-to-speech:** The chatbot can respond with synthesized speech using the `pyttsx3` library.
- **MongoDB integration:** User interactions are stored in a MongoDB database for future reference and training.
- **Machine learning:** The chatbot utilizes scikit-learn to train a Naive Bayes classifier on user interactions for intent recognition.
- **Natural language processing:** The chatbot uses a bag-of-words model for text classification.

## Installation

1. **Clone the repository:**
   
   ```bash
   git clone https://github.com/your_username/chatbot.git
   ```
### Install dependencies:
    ```bash
pip install -r requirements.txt
    ```
#### Set up MongoDB:

Install MongoDB on your local machine or use a cloud-based MongoDB service.
Update the MongoDB connection string in the code (initialize_database() and store_data() functions) to connect to your MongoDB instance.
Run the chatbot:
```bash

python chatbot.py
```
### Usage
When prompted, speak to the chatbot, and it will respond accordingly.
User interactions will be stored in the MongoDB database (chatbot_db).
###### License
This project is licensed under the MIT License - see the LICENSE file for details.