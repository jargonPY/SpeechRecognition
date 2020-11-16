import numpy as np
import os
from glob import glob
import tensorflow as tf
from preprocess_text import PreprocessText

def load_data(path):

    # glob returns a list that match a given pattern (in this case the path name)
    audio_files = glob(os.path.join(path, "filterbanks/*"))
    return audio_files

def read_audio(audio_file):

    audio = np.load(audio_file)
    ## NORMALIZE DATA
    audio = audio.astype(np.float32)
    return audio

def split_dataset(file_names, text, split_ratio=[0.7, 0.15]):

    dataset = tf.data.Dataset.from_tensor_slices((file_names, text))
    # maintains a buffer of buffer_size elements and randomly selects the next element in that buffer
    dataset = dataset.shuffle(buffer_size=len(file_names))

    size = dataset.cardinality().numpy()
    train_ds = dataset.take(size * split_ratio[0])
    remaining = dataset.skip(size * split_ratio[0])  
    val_ds = remaining.take(size * split_ratio[1])
    test_ds = remaining.skip(size * split_ratio[1])

    return train_ds, val_ds, test_ds

def prepare_data(dataset, batch_size=128):

    ## dataset = tf.data.Dataset.from_tensor_slices(({"input_1": sent1, "input_2": sent2}, labels))

    # will apply the preprocessing function on every sample
    dataset = dataset.map(lambda x: read_audio(x[0]))
    dataset = dataset.batch(batch_size)
    # prefetches m elements of its direct input, in this case m batches
    dataset = dataset.prefetch(2)
    return dataset

if __name__ == "__main__":

    path = "/Users/elipalchik/Documents/Programming/python/Signals/"
    text_file = " "
    # create a list of audio_files and a list of the corresponding text
    audio_files = load_data(path)
    process_text = PreprocessText(text_file)
    text_labels = process_text.convert_to_index()
    
    train_ds, val_ds, test_ds = split_dataset(audio_files, text_labels)
    
    train_ds = prepare_data(train_ds)
    val_ds = prepare_data(val_ds, batch_size=val_ds.cardinality().numpy())