import os
import numpy as np
from preprocess_audio import PreprocessAudio
from preprocess_text import PreprocessText

def move_files(path):

    new_audio_path = path + "/audio_files"
    new_text_path = path + "/text_files"
    if not os.path.isdir(new_audio_path):
        os.mkdir(new_audio_path)
    if not os.path.isdir(new_text_path):
        os.mkdir(new_text_path)

    num_files = 0
    for subdir, dirs, files in os.walk(path + "/TRAIN"):
        text_file_name = ''
        for filename in files:
            if ".TXT" in filename:
                text_file_name = filename
            elif ".wav" in filename:
                assert text_file_name.split(".")[0] == filename.split(".")[0]
                
                os.rename(subdir + "/" + filename, new_audio_path + f"/{num_files}.wav")
                os.rename(subdir + "/" + text_file_name, new_text_path + f"/{num_files}.txt")
                num_files += 1

def create_single_text(path):

    new_path = path + "/text_train"
    if not os.path.isdir(new_path):
        os.mkdir(new_path)

    with open(new_path + "/all_text.txt", "w") as f:
        for file in os.listdir(path + "/text_files"):
            with open(path + "/text_files/" + file) as read_file:
                line = read_file.readline().split()
                new_line = " "
                new_line = new_line.join(line[2:])
                f.write(new_line + "\n")
                #f.writelines(lines for line in read_file)

def create_audio_features(path):

    new_path = path + "/audio_train"
    if not os.path.isdir(new_path):
        os.mkdir(new_path)

    for filename in os.listdir(path + "/audio_files"):
        pre = PreprocessAudio(path + "/audio_files/" + filename)
        filter_banks = pre.get_filter_banks().flatten()
        np.save(new_path + f"/{filename}", filter_banks)

def create_text_features(path):

    path = path + "/text_train/all_text.txt"
    if os.path.isfile(path):
        pre = PreprocessText(path)
        pre.process_text()
    else:
        raise ValueError("File does not exist")

if __name__ == "__main__":
    path = "./data"
    move_files(path)
    create_single_text(path)
    create_audio_features(path)
    create_text_features(path)