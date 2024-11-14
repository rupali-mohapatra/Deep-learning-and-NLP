#!/usr/bin/env python
# coding: utf-8

# An end-to-end chatbot refers to a chatbot that can handle a complete conversation from start to finish without requiring human assistance. To create an end-to-end chatbot, you need to write a computer program that can understand user requests, generate appropriate responses, and take action when necessary. This involves collecting data, choosing a programming language and NLP tools, training the chatbot, and testing and refining it before making it available to users.
# 
# ### Steps to follow:
# 1. Define Intents
# 2. Create training data
# 3. Train the chatbot
# 4. Build the chatbot
# 5. Test the chatbot
# 6. Deploy the chatbot

# In[2]:


#!pip install streamlit


# In[3]:


#import necessary libraries
import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


# In[4]:


# 1. Define Intents
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    }
]


# In[9]:


# Styling the Streamlit App
st.markdown(
    """
    <style>
    .stTextInput {
        border-radius: 10px;
        padding: 10px;
        border: 2px solid #4CAF50;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# In[5]:


# Now let's prepare the intents and train a Machine Learning model for the chatbot
# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)


# In[6]:


# Now let’s write a Python function to chat with the chatbot
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response


# In[ ]:


# The chatbot is created now . 
#After running the above code, we can interact with the chatbot in the terminal itself. 
#To turn this chatbot into an end-to-end chatbot, we need to deploy it to interact with the chatbot.
#using an user interface. To deploy the chatbot, I will use the streamlit library, which provides 
#amazing features to create a user interface for a Machine Learning application in just a few lines of code.


# In[10]:


#Delpoying the Chatbot

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

def main():
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

    # User input with unique key
    user_input = st.text_input("You:", key=f"user_input_{len(st.session_state.conversation)}")

    if user_input:
        # Get response from chatbot
        response = chatbot(user_input)
        st.session_state.conversation.append(f"You: {user_input}")
        st.session_state.conversation.append(f"Chatbot: {response}")

    # Display conversation history
    for message in st.session_state.conversation:
        st.text_area(label='', value=message, height=50, key=f"message_{message}", disabled=True)

    # End session if user says goodbye
    if user_input and user_input.lower() in ['goodbye', 'bye']:
        st.write("Thank you for chatting with me. Have a great day!")
        st.session_state.conversation = []  # Reset conversation history
        st.stop()

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




