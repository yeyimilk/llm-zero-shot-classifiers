from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

#Cleaning the texts and getting rid of unclear symbols or characters
#This function is from: https://www.kaggle.com/code/shahraizanwar/covid19-tweets-sentiment-prediction-rnn-85-acc
import re
def text_cleaner(tweet):
    stop_words = stopwords.words('english')
    # remove urls
    tweet = re.sub(r'http\S+', ' ', tweet)  
    # remove html tags
    tweet = re.sub(r'<.*?>',' ', tweet)
    # remove digits
    tweet = re.sub(r'\d+',' ', tweet)    
    # remove hashtags
    tweet = re.sub(r'#\w+',' ', tweet)    
    # remove mentions
    tweet = re.sub(r'@\w+',' ', tweet)    
    #removing stop words
    tweet = tweet.split()
    tweet = " ".join([word for word in tweet if not word in stop_words])   
    return tweet

def data_to_sequence(x_train, x_test, num_words):
    x_train = x_train.apply(text_cleaner)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(x_train)
    
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    
    x_train = pad_sequences(x_train, padding="post")
    x_test = pad_sequences(x_test, padding="post", maxlen=x_train.shape[1])
    
    size = len(tokenizer.word_index) if num_words is None else num_words
    
    return x_train, x_test, size