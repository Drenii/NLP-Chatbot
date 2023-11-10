from flask import Flask, render_template, request
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

f = open('data.txt', 'r', errors='ignore')
raw_doc = f.read().lower()
f.close()

sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

greet_inputs = ('hello', 'hi', 'whassup', 'how are you?')
greet_responses = ('hi', 'Hey', 'Hey There!', 'There there!!')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

def get_chat_response(user_response):
    robo1_response = ''
    TfidVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo1_response = robo1_response + "I'm sorry. Unable to understand you!"
        return robo1_response
    else:
        robo1_response = robo1_response + sentence_tokens[idx]
        return robo1_response

@app.route('/')
def home():
    with open('data.txt', 'r', errors='ignore') as f:
        raw_doc = f.read().lower()

    return render_template('index.html', raw_doc=raw_doc)

@app.route('/get_response', methods=['POST'])
def get_response():
    user_response = request.form['user_input']
    user_response = user_response.lower()

    if user_response != 'bye':
        if user_response == 'thank you' or user_response == 'thanks':
            chat_response = 'You are Welcome..'
        else:
            greeting = greet(user_response)
            if greeting is not None:
                chat_response = greeting
            else:
                sentence_tokens.append(user_response)
                word_tokens.extend(nltk.word_tokenize(user_response))
                final_words = list(set(word_tokens))
                chat_response = get_chat_response(user_response)
                sentence_tokens.remove(user_response)
    else:
        chat_response = 'Goodbye!'

    return chat_response

if __name__ == '__main__':
    app.run(debug=True)
