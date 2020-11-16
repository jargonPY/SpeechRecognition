import numpy as np
import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional, Concatenate

"""
Review of LSTMs:

Input: (num_samples, num_timesteps, num_features) --> output: (num_samples, hidden_dim)

Given time-series data with shape (N x 700) we need to reshape it to (N x 700 x 1). 
Feeding it into an LSTM(200, return_sequences=True) results in an output of shape (N x 700 x 200) --> (num_samples, num_timesteps, num_channels)

You apply a TimeDistributedDense, you're applying a Dense layer on each timestep, which means you're applying a Dense layer on each ℎ1, ℎ2,...,ℎ𝑡 respectively. 
Which means: actually you're applying the fully-connected operation on each of its channels (the "200" one) respectively, from ℎ1 to ℎ700. The 1st "1×1×200" until the 700th "1×1×200".

Why are we doing this? Because you don't want to flatten the RNN output.

Why not flattening the RNN output? Because you want to keep each timestep values separate.

Why keep each timestep values separate? Because:

you're only want to interacting the values between its own timestep
you don't want to have a random interaction between different timesteps and channels.

RepeatVector: repeats output of previous layer a specified number of times
TimeDistributed: applies a dense layer to every sample

Return_sequences vs. Return_state: 
LSTM(dim, return_sequences=True) --> returns hidden state for each time step
LSTM(dim, return_state=True) --> returns LSTM hidden state output (twice) and LSTM cell state for the last time step
LSTM(dim, return_sequences=True, return_state=True) --> returns hidden state for each time step as well as the hidden and cell state for last output
 - when stacking LSTMS return_sequences must be True
"""

""" ENCODER - DECODER ARCHITECTURE """

class ListenAttendSpell():

    def __init__(self, batch_size=64, epochs=128, hidden_dim=256):

        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_dim = hidden_dim

        self.num_tokens = 26
        self.max_length = 100

    def pBLSTM(self, input_tensor, name=''):

        blstm = Bidirectional(LSTM(self.hidden_dim, return_sequences=True, name=name))
        _, forward_h, back_h = blstm(input_tensor)
       # state_c = Concatenate()([forward_c, back_c])
        state_h = Concatenate()([forward_h, back_h])
       # state_h = Concatenate()([state_h, state_h]) #### RESHAPE
        return state_h

    def build_model(self, num_layers=3, embed_dim=256):

        # Input is a 1D sequence consisting of the filter banks derived from the audio signal
        enc_input = Input(shape=(None, 1))
        enc_input = keras.layers.Masking(mask_value=0., input_shape=(None, 1))(enc_input)

        for i in range(num_layers):
            if i == 0:
                listen_h, listen_c = self.pBLSTM(enc_input, name=f"encoder_layer{i}")
            else:
                listen_h, listen_c = self.pBLSTM(listen_h, name=f"encoder_layer{i}")
        
        # Input is a 1D sequence consisting of the padded text sentence
        dec_input = Input(shape=(self.max_length, 1))
        embed = Embedding(self.num_tokens, embed_dim, input_length=self.max_length)(dec_input)
        _, embed_h, embed_c = LSTM(self.hidden_dim, return_sequences=True)(embed)
        # inputs: (batch_size, Tq, dim) and (batch_size, Tv, dim) || output: (batch_size, Tq, dim)
        attention = keras.layers.AdditiveAttention()([embed_h, listen_h])

        _, state_h, state_c = LSTM(self.hidden_dim, return_sequences=True)(attention)
        output = Dense(self.max_length, activation='softmax')(state_h)

        model = keras.models.Model(inputs=[enc_input, dec_input], outputs=output)
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self, train_ds, val_ds, file_path, epochs=128):

        earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(file_path, monitor="val_accuracy", save_best_only=True)

        self.model.fit(train_ds,
                       epochs=epochs,
                       validation_data=val_ds,
                       callbacks=[earlystopping_cb, mdlcheckpoint_cb])

    def save_model(self, file_path):

        try:
            self.model.save(file_path)
        except Exception as e:
            raise ValueError(e)

    def load_model(self, file_path):

        try:
            self.model = keras.models.load_model(file_path)
        except Exception as e:
            raise ValueError(e)


model = ListenAttendSpell()
model.build_model()
model.model.summary()
