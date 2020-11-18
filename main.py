import random
import os
from models.listen_attend_spell import ListenAttendSpell
from utils.preprocess_text import PreprocessText
from utils.data_generator import DataGenerator

def get_audio_files(path):

    def sort_by_number(value):
        return int(value.split(".")[0])

    # glob returns a list that match a given pattern (in this case the path name)
    audio_files = os.listdir(path)
    audio_files = [f for f in audio_files if '.npy' in f]
    audio_files.sort(key=sort_by_number)
    return audio_files

def split_dataset(audio_files, text, split_ratio=0.85):

    ids = dict([(file_name, text) for file_name, text in zip(audio_files, text)])
    random.shuffle(audio_files)

    train_files = audio_files[:int(len(audio_files) * split_ratio)]
    val_files =  audio_files[int(len(audio_files) * split_ratio):]

    train_ids = dict([(file_name, ids[file_name]) for file_name in train_files])
    val_ids = dict([(file_name, ids[file_name]) for file_name in val_files])
    return train_ids, val_ids

def get_data():

    path = "./data/audio_train"
    text_file = "./data/text_train/all_text.txt"
    # create a list of audio_files and a list of the corresponding text
    audio_files = get_audio_files(path)
    preprocess = PreprocessText(text_file)
    input_text, output_text = preprocess.process_text()
    text = [(in_line, out_line) for in_line, out_line in zip(input_text, output_text)]
    return audio_files, text, preprocess.num_tokens

def generate_train_val(audio_files, text, num_tokens):

    # split data into train/validation/test sets
    train_ids, val_ids = split_dataset(audio_files, text)
    # prepare batches for training
    train_ds = DataGenerator(train_ids, num_classes=num_tokens)
    val_ds = DataGenerator(val_ids, num_classes=num_tokens, batch_size=len(val_ids))
    return train_ds, val_ds

def generate_model(num_tokens):

    model = ListenAttendSpell(num_classes=num_tokens)
    model.build_model()
    return model

if __name__ == "__main__":

    audio_files, text, num_tokens = get_data()
    train_ds, val_ds = generate_train_val(audio_files, text, num_tokens)
    model = generate_model(num_tokens)
    model.train(train_ds, val_ds, "./models/saved_models/checkpoint")