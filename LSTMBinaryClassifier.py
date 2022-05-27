import numpy as np

np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Bidirectional
import copy
from keras import backend as K
from keras.layers import Flatten,Permute,RepeatVector,Multiply,Lambda

np.random.seed(1)

class Attention:
    def __call__(self, inp, combine=True, return_attention=True):
        # Expects inp to be of size (?, number of words, embedding dimension)
        
        repeat_size = int(inp.shape[-1])  #512
        
        # Map through 1 Layer MLP
        x_a = Dense(repeat_size, kernel_initializer = 'glorot_uniform', activation="tanh", name="tanh_mlp")(inp) #none, 52, 512
        
        # Dot with word-level vector
        x_a = Dense(1, kernel_initializer = 'glorot_uniform', activation='linear', name="word-level_context")(x_a) #none, 52, 1
        x_a = Flatten()(x_a) # x_a is of shape (?,52,1), we flatten it to be (?,52)
        att_out = Activation('softmax')(x_a) 
        
        # Clever trick to do elementwise multiplication of alpha_t with the correct h_t:
        # RepeatVector will blow it out to be (?,52, 512)
        # Then, Permute will swap it to (?,52,512) where each row (?,k,120) is a copy of a_t[k]
        # Then, Multiply performs elementwise multiplication to apply the same a_t to each
        # dimension of the respective word vector
        x_a2 = RepeatVector(repeat_size)(att_out) #none, 512, 52
        x_a2 = Permute([2,1])(x_a2) #none, 52, 512
        out = Multiply()([inp,x_a2]) #none, 52, 512
        
        if combine:
        # Now we sum over the resulting word representations
            out = Lambda(lambda x : K.sum(x, axis=1), name='expectation_over_words')(out) #none, 512
        
        if return_attention:
            out = (out, att_out)
                   
        return out

def LSTMModel(maxL,vectorLength):
    net_input = Input(shape=(maxL,vectorLength))
    lstm = Bidirectional(LSTM(256, return_sequences=True))(net_input)
    x, attention = Attention()(lstm)
    dense = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=net_input, outputs=dense)
    track = Model(inputs=net_input, outputs=attention)
 
    return model,track;


