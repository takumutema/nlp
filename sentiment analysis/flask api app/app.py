import string
import pickle
import re
from numpy import array
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model


from flask import Flask
from flask import request
import json

app = Flask(__name__)

# load the model
model = load_model('model.h5', compile=False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# load doc into memory
def load_doc(filename):
 # open the file as read only
 file = open(filename, 'r')
 # read all text
 text = file.read()
 # close the file
 file.close()
 return text

# turn a doc into clean tokens
def clean_doc(doc, vocab):
  # split into tokens by white space
  tokens = doc.split()
  # prepare regex for char filtering
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))
  # remove punctuation from each word
  tokens = [re_punc.sub('', w) for w in tokens]
  # filter out tokens not in vocab
  tokens = [w for w in tokens if w in vocab]
  tokens = ' '.join(tokens)
  return tokens

# fit a tokenizer
def create_tokenizer(lines):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(lines)
  return tokenizer

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
  # integer encode
  encoded = tokenizer.texts_to_sequences(docs)
  # pad sequences
  padded = pad_sequences(encoded, maxlen=max_length, padding='post')
  return padded

# classify a review as negative or positive
def predict_sentiment(review, vocab, tokenizer, max_length, model):
  # clean review
  line = clean_doc(review, vocab)
  # encode and pad review
  padded = encode_docs(tokenizer, max_length, [line])
  # predict sentiment
  yhat = model.predict(padded, verbose=0)
  # retrieve predicted percentage and label
  percent_pos = yhat[0,0]
  if round(percent_pos) == 0:
    return str(1-percent_pos), 'NEGATIVE'
  return str(percent_pos), 'POSITIVE'

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# Load the list of train_docs from a file
with open('train_docs.pkl', 'rb') as f:
    train_docs = pickle.load(f)

# create the tokenizer
tokenizer = create_tokenizer(train_docs)
# define vocabulary size
vocab_size = len(tokenizer.word_index) + 1
# calculate the maximum sequence length
max_length = max([len(s.split()) for s in train_docs])


@app.route('/api', methods=['GET','POST'])
def handle_request():
    text = str(request.args.get('review')) #requests the ?review=''
    if text == 'None':
      return "API web page goes here"
    else:
      percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
      data_set = {'text': text , 'sentiment':sentiment, 'percentage':percent}
      json_dump = json.dumps(data_set)
      return json_dump

if __name__ == "__main__":
  app.run(host = "0.0.0.0", port = int("5000"), debug=True)