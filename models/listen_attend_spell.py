import numpy as np
import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Concatenate

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

    def __init__(self, num_classes=26, max_sent_len=100, hidden_dim=256):

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_sent_len = max_sent_len

    def pBLSTM(self, input_tensor, name):
        # if merge_mode = "concat" --> embed_h must be 2 * self.hidden_dim
        blstm = Bidirectional(LSTM(self.hidden_dim, return_sequences=True), merge_mode="ave", name=name)
        state_h = blstm(input_tensor)
        # state_h = Concatenate()([state_h, state_h]) #### RESHAPE
        return state_h

    def encoder(self, num_layers=3):

        # Input is a 1D sequence consisting of the filter banks derived from the audio signal
        encoder_input = Input(shape=(None, 1), name="encoder_input")
        listen_h = keras.layers.Masking(mask_value=0., input_shape=(None, 1))(encoder_input)

        for i in range(num_layers):
            if i == (num_layers - 1):
                listen_h = self.pBLSTM(listen_h, name="encoder_output")
            else:
                listen_h = self.pBLSTM(listen_h, name=f"encoder_layer{i}")

        self.encoder_model = keras.models.Model(inputs=encoder_input, outputs=listen_h, name='Encoder')

    def decoder(self, embed_dim=256):

        # Input is a 1D sequence consisting of the padded text sentence
        decoder_input = Input(shape=(None, ), name="decoder_input")
        encoder_output = Input(shape=(None, self.hidden_dim))
        embed = Embedding(self.num_classes, embed_dim, mask_zero=True, name="dec_embed")(decoder_input)
        embed_h = LSTM(self.hidden_dim, return_sequences=True, name="LSTM_embed")(embed)
        # inputs: (batch_size, Tq, dim) and (batch_size, Tv, dim) || output: (batch_size, Tq, dim)
        attention = keras.layers.AdditiveAttention(name="attention")([embed_h, encoder_output])

        state_h = LSTM(self.hidden_dim, return_sequences=True, name="LSTM_attend")(attention)
        decoder_output = Dense(self.num_classes, activation='softmax', name="decoder_output")(state_h)
    
        self.decoder_model = keras.models.Model(inputs=[decoder_input, encoder_output], outputs=decoder_output, name='Decoder')
    
    def build_model(self):

        input1 = Input(shape=(None, 1), name="input_1")
        encoder_output = self.encoder_model(input1)
        input2 = Input(shape=(None, ), name="input_2")
        decoder_output = self.decoder_model([input2, encoder_output])

        model = keras.Model(inputs=[input1 ,input2], outputs=decoder_output, name="LAS")
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self, train_ds, val_ds, file_path, batch_size=128, epochs=128):

        earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(file_path, monitor="val_accuracy", save_best_only=True)

        self.model.fit(train_ds,
                       epochs=epochs,
                       validation_data=val_ds,
                       callbacks=[earlystopping_cb, mdlcheckpoint_cb],
                       verbose=1)

    def inference(self, audio, token_to_index, index_to_token):

        encoder_output = self.encoder_model.predict(audio)

        # generate empty target sequence of length 1 with only the start character
        target_seq = np.zeros((1, 1, self.num_classes))
        target_seq[0, 0, token_to_index["<sos>"]] = 1.
        
        # output sequence loop
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + encoder_output)
            # sample a token and add the corresponding character to the 
            # decoded sequence
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = index_to_token[sampled_token_index]
            decoded_sentence += sampled_char
            
            # check for the exit condition: either hitting max length
            # or predicting the 'stop' character
            if (sampled_char == "<eos>" or len(decoded_sentence) > self.max_sent_len):
                stop_condition = True
            
            # update the target sequence (length 1).
            target_seq = np.zeros((1, 1, self.num_classes))
            target_seq[0, 0, sampled_token_index] = 1.
            
            # update states
            encoder_output = [h, c]
            
        return decoded_sentence
    
    def load_model(self, file_path):

        try:
            self.model = keras.models.load_model(file_path)
        except Exception as e:
            raise ValueError(e)