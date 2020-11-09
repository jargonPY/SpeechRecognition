import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sounddevice as sd
import scipy.io.wavfile as wavefile
from scipy.signal import spectrogram
import time
from scipy.signal import butter, lfilter, freqz, firwin

"""
fs = 44100
t = 4
print("Recording Now")
data = sd.rec(frames=t * fs, samplerate=44100, channels=2, blocking=True)
print("Recording Stopped")
print(data)
sd.play(data, fs, blocking=True)
wavefile.write('./data.wav', fs, data)
"""

class SplitSentence():

    def __init__(self, signal, fs=44100, win_size=300):
        
        self.signal = signal[0:50000, 1]
        self.len = len(self.signal)
        self.win_size = win_size
        self.fs = fs

    def rolling_window(self, signal, window=256, shift=128):
    
        shape = signal.shape[:-1] + (int(signal.shape[-1] - window + 1), int(window))
        strides = signal.strides + (signal.strides[-1],)
        return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)[::shift]

    def normalize(self, signal):

        norm = np.empty_like(signal)
        for i in range(0, len(signal), self.win_size):
            interval = signal[i:i + self.win_size]
            norm[i:i + self.win_size] = (interval - np.mean(interval)) / np.std(interval)
        return norm

    def smooth(self, window='hanning'):

        s = np.r_[self.signal[self.win_size - 1:0:-1], self.signal, self.signal[-2:-self.win_size - 1:-1]]
        
        if window == 'flat': #moving average
            w = np.ones(self.win_size, 'd')
        else:
            w = eval('np.'+window+'(self.win_size)')

        filtered_sig = np.convolve(w / w.sum(), s, mode='valid')
        return filtered_sig

    def fir(self, cutoff=[400, 4000], stop=False):
        """
        stop == False --> high-pass / band-pass
        stop == True --> low-pass / band-stop
        """

        if type(cutoff) == list:
            cutoff = [i / (self.fs / 2) for i in cutoff]
        else:
            cutoff /= (self.fs / 2)
        # finds the coefficients of a FIR filter
        coeff = firwin(self.win_size, cutoff, window='hamming', pass_zero=stop)
        # use lfilter to filter x with the FIR filter.
        filtered_sig = lfilter(coeff, 1.0, self.signal)
        return filtered_sig

    def butter_bandpass(self, lowcut, highcut, order=5):

        nyq = int(self.fs / 2)
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, lowcut, highcut, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, order=order)
        filtered_sig = lfilter(b, a, self.signal)
        return filtered_sig

    def mean(self, signal):

        mean = [ ]
        for i in range(0, len(signal), self.win_size):
            mean.append(np.absolute(np.mean(signal[i:i + self.win_size])))
        return mean

    def std(self, signal):
        
        std = [ ]
        for i in range(0, len(signal), self.win_size):
            std.append(np.std(signal[i:i + self.win_size]))
        return std


    def convolve(self, signal):

        convolve = [ ]
        for i in range(0, len(signal) - 2 * self.win_size, self.win_size):
            conv = np.absolute(np.convolve(signal[i:i + self.win_size], signal[i + self.win_size:i + 2 * self.win_size]))
            mean = np.mean(conv)
            if mean < 0.01:
                mean = 0
            convolve.append(mean)
        return convolve

    def split_signal(self, param):

        threshold = 0.002
        val = [ ]
        for i in range(2, len(param)):
            prev = param[i-1]
            prev2 = param[i-2]
            if (param[i] > threshold and prev < threshold and prev2 < threshold):
                val.append(i)
        
        split = [i * self.win_size for i in val]
        return split

    def filters(self, cutoff, filter_type):

        if filter_type == 'smooth':
            new_sig = self.smooth()
        elif filter_type == 'highpass':
            new_sig = self.fir(cutoff=cutoff[0])
        elif filter_type == 'lowpass':
            new_sig = self.fir(cutoff=cutoff[0], stop=True)
        elif filter_type == 'bandpass':
            new_sig = self.fir(cutoff=[cutoff[0], cutoff[1]])
        elif filter_type == 'bandstop':
            new_sig = self.fir(cutoff=[cutoff[0], cutoff[1]], stop=True)
        elif filter_type == 'butter':
            new_sig = self.butter_bandpass_filter(cutoff[0], cutoff[1])
        return new_sig

    def single_filter(self, cutoff, filter_type, properties=['mean', 'std', 'convolve'], split_on='std', norm=False):

        if norm:
            self.signal = self.normalize(self.signal)

        new_sig = self.filters(cutoff, filter_type=filter_type)

        stats = {'new_sig': new_sig}
        for prop in properties:
            stats[prop] = eval('self.' + prop + '(new_sig)')
        split = self.split_signal(stats[split_on])
        return stats, split

    def multi_filter(self, filters, norm=False):

        if norm:
            self.signal = self.normalize(self.signal)

        filtered = { }
        for key in filters.keys():
            stats, split = self.single_filter(filters[key], filter_type=key)
            filtered[key] = [stats['new_sig'], split]

        self.multi_filter_plot(filtered)

    def single_filter_plot(self, stats, split):
        
        fig, axs = plt.subplots(len(stats), 1)
        # plot the signal
        axs[0].plot(range(self.len), self.signal, label='original')
        axs[0].plot(range(len(stats['new_sig'])), stats['new_sig'], label='filtered')
        axs[0].vlines(split, ymin=min(self.signal), ymax=max(self.signal), linestyles='dashed')
        axs[0].set_title('Signal')
        axs[0].legend()
        # plot statistical properties of the signal
        keys = list(stats.keys())
        keys.remove('new_sig')
        for i, key in enumerate(keys):
            axs[i+1].stem(range(len(stats[key])), stats[key])
            axs[i+1].set_title(f'{key}')
        plt.show()

    def multi_filter_plot(self, filters):

        fig, axs = plt.subplots(len(filters), 1)
        for i, key in enumerate(filters.keys()):
            new_sig = filters[key][0]
            split = filters[key][1]
            axs[i].plot(range(self.len), self.signal, label='original')
            axs[i].plot(range(len(new_sig)), new_sig, label='filtered')
            axs[i].vlines(split, ymin=min(self.signal), ymax=max(self.signal), linestyles='dashed')
            axs[i].set_title(f'{key}')
            axs[i].legend()
        plt.show()

    def spectrogram(self, seg_len=256, overlap_len=128):

        f, t, Sxx = spectrogram(self.signal, fs=self.fs, window='hamming', nperseg=seg_len, noverlap=overlap_len)
        # Sxx = (frequency, time) --> coloumns represent frequencies at a given time
        Sxx = 10*np.log10(Sxx)

        fig, axs = plt.subplots(1, 1)
        axs.pcolormesh(t, f, Sxx, shading='gouraud')
        axs.set_xlabel('Time [sec]')
        axs.set_ylabel('Frequency [Hz]')
        plt.show()


if __name__ == "__main__":
    rate, signal = wavefile.read('./data.wav')
    pre = SplitSentence(signal, win_size=300)
    #new_sig, avg, std, split = split.single_filter(filter_type='smooth')
    #new_sig, avg, std, split = pre.single_filter(1000, 4500, filter_type='bandpass')
    #stats, split = pre.single_filter([1000, 4500], filter_type='bandpass')
    #pre.single_filter_plot(stats, split)
    
    filters = {'smooth':[], 'bandpass':[1000, 4500], 'butter':[1000, 4500]}
    pre.multi_filter(filters)