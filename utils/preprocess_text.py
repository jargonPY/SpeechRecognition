import numpy as np
import os
import string

class PreprocessText():

    def __init__(self, file_name, one_hot_encoding=False):

        self.file_name = file_name
        self.text = [ ]
        self.tokens = set()
        self.num_tokens = None
        self.max_length = None
        self.token_to_index = None
        self.index_to_token = None

    def unique_chars(self):

        with open(self.file_name) as f:
            for line in f:
              #  line = line.split('(')[0][:-1] # remove end of line space
                line = self.clean_text(line)
                for char in line:
                    self.tokens.add(char)
                line = [char for char in line]
                line.insert(0, '<sos>')
                line.append('<eos>')
                self.text.append(line)

    def add_chars(self):

        special_tokens = ['<sos>', '<eos>', ] ### ---> WHEN IS ZERO APPENDED?
        for i in special_tokens:
            self.tokens.add(i)

    def get_values(self):

        self.num_tokens = len(self.tokens)
        self.max_length = max([len(txt) for txt in self.text])
        self.token_to_index =  dict([(char, i) for i, char in enumerate(self.tokens)])
        self.index_to_token = dict([(i, char) for i, char in enumerate(self.tokens)])

    def parse_file(self):

        self.add_chars()
        self.unique_chars()
        self.get_values()

    def convert_to_index(self):

        padded = np.zeros((len(self.text), self.max_length))
        for i, phrase in enumerate(self.text):
            for j, char in enumerate(phrase):
                    padded[i][j] = self.token_to_index[char]
        return padded

    def one_hot_encoding(self, padded):

        decoder_input = np.zeros((len(self.text), self.max_length, len(self.tokens)))
        decoder_target = np.zeros((len(self.text), self.max_length, len(self.tokens)))
        for i, phrase in enumerate(padded):
            for j, char in enumerate(phrase):
                decoder_input[i, j, self.token_to_index[char]] = 1.0
                if j > 0:
                    decoder_target[i, j - 1, self.token_to_index[char]] = 1.0
        return decoder_input, decoder_target

    def process_text(self):

        self.parse_file()
        padded = self.convert_to_index()
        input_text = padded[:, :-1]
        output_text = padded[:, 1:]
        return input_text, output_text

    @staticmethod
    def clean_text(line):

        line = line.translate(str.maketrans('', '', string.punctuation))
        line = line.lower()
        return line