import tensorflow as tf

import tensorflow.keras.layers as layer
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from utils import save_to_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils import setup_device
from utils import NpEncoder
import json
from data_loader import get_data_by_name

random_state = 100
np.random.seed(random_state)
tf.random.set_seed(random_state)

# https://github.com/lorrespz/NLP-text-analysis
def create_generice_model(model_layer, units, input_length, outs, vocab_size):
    setup_device()
    if tf.config.experimental.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy() 
    
    embedding_dim = 16

    with strategy.scope():
        model= tf.keras.Sequential([
            layer.Embedding(vocab_size, embedding_dim, input_length=input_length), # X.shape[1]
            model_layer(units,return_sequences = True),
            layer.GlobalMaxPooling1D(),
            layer.Dense(outs, activation = 'softmax')
        ])
        
        model.compile(loss=SparseCategoricalCrossentropy(from_logits = False),
                optimizer='adam',metrics=['accuracy'])
    return model


def generate_reports(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr') if len(set(y_true)) > 2 else roc_auc_score(y_true, y_pred),
    
    result = {
        "acc": acc,
        "f1": f1,
        "conf": confusion_matrix(y_true, y_pred),
        "auc": auc[0]
    }
    print(f"========{name}=========")
    print(f"{acc} & {f1} & {auc[0]}")
    print(json.dumps(result, cls=NpEncoder))
    return result

def train_model(model, x_train, y_train, x_test, y_test, name, epochs, batch_size):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
    history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        validation_split=0.15, 
                        batch_size=batch_size,
                        callbacks = [callback])
    hist = pd.DataFrame(history.history)
    pred = np.argmax(np.round(model.predict(x_test)), axis = 1)
    pred_d = model.predict(x_test)
    pred = np.argmax(np.round(pred_d), axis = 1)
    
    reports = generate_reports(name, y_test, pred, pred_d)
    
    result = {
        "reports": reports,
        "y_test": y_test,
        "y_pred": pred, 
        "y_pred_d": pred_d
    }
        
    fpath = save_to_file(result, name, type='json')
    hist.to_csv(fpath.replace('.json', '_hist.csv'), index=False)
    
    return result

def train_nns(x_train, y_train, x_test, y_test, outs, name, vocab_size):
    rnn = create_generice_model(layer.SimpleRNN, 20, x_train.shape[1], outs, vocab_size)
    lstm = create_generice_model(layer.LSTM, 15, x_train.shape[1], outs, vocab_size)
    gru = create_generice_model(layer.GRU, 15, x_train.shape[1], outs, vocab_size)
    
    train_model(rnn, x_train, y_train, x_test, y_test, f"{name}_rnn", 5, 32)
    train_model(lstm, x_train, y_train, x_test, y_test, f"{name}_lstm", 8, 32)
    train_model(gru, x_train, y_train, x_test, y_test, f"{name}_gru", 6, 32)


def train_classical(x_train, y_train, x_test, y_test, name):
    models = [MultinomialNB(force_alpha=True), LogisticRegression(), 
              RandomForestClassifier(), DecisionTreeClassifier(), 
              KNeighborsClassifier()]
    results = {}
    for model in models:
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        
        model_name = model.__class__.__name__
        pred = model.predict(x_test)
        pred_d = model.predict_proba(x_test)
        reports = generate_reports(model_name, y_test, pred, pred_d)
        
        result = {
            "reports": reports,
            "y_test": y_test,
            "y_pred": pred, 
            "y_pred_d": pred_d
        }
        
        results[model_name] = result
    save_to_file(result, f"{name}_ml", type='json')
    
def train(name, outs):
    print(f"Start training {name}")
    num_words = None
    x_train, y_train, x_test, y_test, size = get_data_by_name(name, num_words)
    train_classical(x_train, y_train, x_test, y_test, name)
    train_nns(x_train, y_train, x_test, y_test, outs, name, size+1)

if __name__ == '__main__':
    names = ['Corona_NLP_1', 'ecommerceDataset', 'sms_spam', 'financial_sentiment']
    n_outs = [3, 4, 2, 3]
    for i in range(len(names)):
        train(names[i], n_outs[i])