import tensorflow as tf
import spacy 
import pandas as pd
import numpy as np
import LSTMBinaryClassifier as lsc
from keras.utils import Sequence
import os
import math
from Preprocessing import get_embeddings
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

class TripletGenereator(Sequence):
    def __init__(self, x, y, batch_size=32, max_length=20, embedding_size=1100):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedding_size = embedding_size
        self.indices = np.arange(len(self.x))
        np.random.shuffle(self.indices)
        self.validation_accuracy = []

    def __len__(self):
        return math.ceil(float(len(self.y)) / self.batch_size)

    def __getitem__(self, idx):
        idxes = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = np.zeros((len(idxes), self.max_length, self.embedding_size))
        y_batch = np.zeros((len(idxes),))
        for i, idx in enumerate(idxes):
            x = self.x[idx]
            if x.shape[0] < self.max_length:
                x = np.vstack(
                    [x, np.zeros((self.max_length - x.shape[0], self.embedding_size))])
            else:
                x = x[:self.max_length, :]
            x_batch[i] = x
            y_batch[i] = self.y[idx]
        return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

get_embeddings("train")
get_embeddings("test")
get_embeddings("val")

max_length = 40
hidden_size = 256
batch_size = 128
EMBEDDING_SIZE = 1100

file_path = os.path.abspath("")

with open(os.path.join(file_path, "dataset", "train_x.pkl"), "rb") as fx:
    x_train = pickle.load(fx)
with open(os.path.join(file_path, "dataset", "train_y.pkl"), "rb") as fy:
    y_train = pickle.load(fy)
with open(os.path.join(file_path, "dataset", "train_paths_x.pkl"), "rb") as fz:
    path_train = pickle.load(fz)

with open(os.path.join(file_path, "dataset", "test_x.pkl"), "rb") as fx:
    x_test = pickle.load(fx)
with open(os.path.join(file_path, "dataset", "test_y.pkl"), "rb") as fy:
    y_test = pickle.load(fy)
with open(os.path.join(file_path, "dataset", "test_paths_x.pkl"), "rb") as fz:
    path_test = pickle.load(fz)

with open(os.path.join(file_path, "dataset", "val_x.pkl"), "rb") as fx:
    x_val = pickle.load(fx)
with open(os.path.join(file_path, "dataset", "val_y.pkl"), "rb") as fy:
    y_val = pickle.load(fy)
with open(os.path.join(file_path, "dataset", "val_paths_x.pkl"), "rb") as fz:
    path_val = pickle.load(fz)

print(len(x_train))
print(len(y_train))
print(len(path_train))

print(len(x_test))
print(len(y_test))
print(len(path_test))


test_sentences =  pd.read_csv(os.path.join(file_path, "dataset", "test_dataset.csv")).values


x_test_words = test_sentences[:,0]
y_test_words = test_sentences[:,1]
test_sentences = test_sentences[:,2]

print(len(x_test_words))
print(len(y_test_words))
print(len(test_sentences))


train_generator = TripletGenereator(
    x_train, y_train, batch_size=batch_size, max_length=max_length, embedding_size=EMBEDDING_SIZE)
validation_generator = TripletGenereator(
    x_val, y_val, batch_size=batch_size, max_length=max_length, embedding_size=EMBEDDING_SIZE)
test_generator = TripletGenereator(
    x_test, y_test, batch_size=batch_size, max_length=max_length, embedding_size=EMBEDDING_SIZE)

model,attention = lsc.LSTMModel(max_length,EMBEDDING_SIZE)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

mc = ModelCheckpoint(os.path.join(file_path, "model", "best_old_model.h5"), monitor="val_loss",verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', patience=3,verbose=1, restore_best_weights=True)

class_weight = {0: 1.,
                1: 1.}

hs = model.fit_generator(train_generator, validation_data=validation_generator,epochs=10000, callbacks=[mc, es],class_weight=class_weight)

train_accuracies = hs.history["acc"]
val_accuracies = hs.history["val_acc"]
train_loss = hs.history["loss"]
val_loss = hs.history["val_loss"]
Epochs = hs.epoch

x_test_with_emb = np.zeros((len(x_test), max_length, EMBEDDING_SIZE))
for i, xt in enumerate(x_test):
    if xt.shape[0] < max_length:
        xt = np.vstack([xt, np.zeros((max_length - xt.shape[0], EMBEDDING_SIZE))])
    else:
        xt = xt[:max_length, :]
    x_test_with_emb[i] = xt

result = attention.predict(x_test_with_emb)

att = [["{:.25f}".format(float(b)) for b in p] for p in result]
att = [",".join(a) for a in att]
path_test = [[list(b) for b in a] for a in list(path_test)]
path_test1 = ['%,,%'.join(['!!'.join(b) for b in a]) for a in path_test]
path_test2 = ['!'.join([','.join(b) for b in a]) for a in path_test]

pred = model.predict(x_test_with_emb)
pred = [a[0] for a in pred]

df = pd.DataFrame(data = {'path': path_test1, 'attention': att, 'label': y_test, 'prediction': pred})

df.to_csv("test_result.csv",index = None,columns = None, header = None)

df = pd.DataFrame(data = {'x': x_test_words, 'y': y_test_words, 'sentences':test_sentences, 'path': path_test2, 'attention': att, 'label': y_test, 'prediction': pred})

pred = []
for a in df.values[:,-1]:
    if(a>0.5):
        pred.append(1)
    else:
        pred.append(0)
    
label = df.values[:,-2]

label = [int(a) for a in label]

print("Test accuracy: "+str(accuracy_score(label, pred)))
print("Precision: "+str(precision_score(label, pred)))
print("Recall: "+str(recall_score(label, pred)))
print("F1 score: "+str(f1_score(label, pred)))

df.to_csv("test_result_all.csv",index = None,columns = None, header = None)

plt.plot(train_accuracies, color = 'blue', linestyle='dashed')
plt.plot(val_accuracies,  color = 'blue')
plt.plot(train_loss,  color='green', linestyle='dashed')
plt.plot(val_loss,  color = 'green')
plt.show()
i=0

