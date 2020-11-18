import os
import numpy as np
from utils.preprocess_audio import PreprocessAudio
from utils.preprocess_text import PreprocessText

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

    files_to_delete = [ ]
    with open(new_path + "/all_text.txt", "w") as f:
        for filename in os.listdir(path + "/text_files"):
            with open(path + "/text_files/" + filename) as read_file:
                line = read_file.readline()
                if ";" in line:
                    files_to_delete.append(filename.split(".")[0])
                    continue
                line = line.split()
                new_line = " "
                new_line = new_line.join(line[2:])
                f.write(new_line + "\n")
    return files_to_delete
                
def create_audio_features(path, files_to_delete):

    new_path = path + "/audio_train"
    if not os.path.isdir(new_path):
        os.mkdir(new_path)

    for filename in os.listdir(path + "/audio_files"):
        if filename.split(".")[0] in files_to_delete:
            continue
        pre = PreprocessAudio(path + "/audio_files/" + filename)
        filter_banks = pre.get_filter_banks().flatten()
        np.save(new_path + f"/{filename}", filter_banks)

if __name__ == "__main__":
    path = "./data"
    move_files(path)
    files_to_delete = create_single_text(path)
    create_audio_features(path, files_to_delete)