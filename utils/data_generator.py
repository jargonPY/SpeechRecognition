import numpy as np
import os
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):

    def __init__(self, ids, batch_size=128, num_classes=26, shuffle=True, path="./data/audio_train"):

        self.ids = ids
        self.list_ids = [key for key in ids.keys()]
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.path = path
        self.on_epoch_end()

    def _read_audio(self, audio_file):

        audio = np.load(self.path + "/" + audio_file)
        ## NORMALIZE DATA
        audio = audio.astype(np.float32)
        return audio

    def _generate_data(self, batch_ids):

        audio_input = [ ]
        text_input = [ ]
        text_output = [ ]

        for id_ in batch_ids:
            audio = self._read_audio(id_)
            audio_input.append(audio)
            text_input.append(self.ids[id_][0])
            out_text = keras.utils.to_categorical(self.ids[id_][1], num_classes=self.num_classes)
            text_output.append(out_text)

        audio_input = keras.preprocessing.sequence.pad_sequences(audio_input, padding="post")
        audio_input = np.asarray(audio_input, np.float32)
        text_input = np.asarray(text_input, np.float32)
        text_output = np.asarray(text_output, np.float32)
        return audio_input, text_input, text_output

    def on_epoch_end(self):

        self.indices = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):

        return int(np.floor(len(self.ids)) / self.batch_size)

    def __getitem__(self, index):

        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_ids = [self.list_ids[i] for i in indices]
        audio_input, text_input, text_output = self._generate_data(batch_ids)
        return [audio_input, text_input], text_output