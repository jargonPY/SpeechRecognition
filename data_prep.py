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

    def normalize(self, signal):

        norm = np.empty_like(signal)
        for i in range(0, len(signal), self.win_size):
            interval = signal[i:i + self.win_size]
            norm[i:i + self.win_size] = (interval - np.mean(interval)) / np.std(interval)
        return norm

    def smooth(self, window='hanning'):
        """
        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
        output:
            the smoothed signal
            
        Note: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """

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
        # Use lfilter to filter x with the FIR filter.
        filtered_sig = lfilter(coeff, 1.0, self.signal)
        return filtered_sig

    def stats(self, signal):

        avg = [ ]
        std = [ ]
        for i in range(0, len(signal), self.win_size):
            interval = signal[i:i + self.win_size]
            avg.append(np.absolute(np.mean(interval)))
            std.append(np.std(interval))
        return avg, std

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
        cutoff = [ ]
        for i in range(2, len(param)):
            prev = param[i-1]
            prev2 = param[i-2]
            if (param[i] > threshold and prev < threshold and prev2 < threshold):
                cutoff.append(i)
        
        x_axis = [i * self.win_size for i in cutoff]
        return cutoff, x_axis

    def filters(self, *args, filter_type='smooth'):

        if filter_type == 'smooth':
            new_sig = self.smooth()
        elif filter_type == 'highpass':
            new_sig = self.fir(cutoff=args[0])
        elif filter_type == 'lowpass':
            new_sig = self.fir(cutoff=args[0], stop=True)
        elif filter_type == 'bandpass':
            new_sig = self.fir(cutoff=[args[0], args[1]])
        elif filter_type == 'bandstop':
            new_sig = self.fir(cutoff=[args[0], args[1]], stop=True)
        return new_sig
        
    def plot(self, *args, norm=True, filter_type='smooth'):
        
        if norm:
           self.signal = self.normalize(self.signal)

        new_sig = self.filters(*args, filter_type=filter_type)
        avg, std = self.stats(new_sig)
        conv = self.convolve(new_sig)

        cutoff, x_axis = self.split_signal(std)

        fig, axs = plt.subplots(4, 1)
        axs[0].plot(range(self.len), self.signal, label='original')
        axs[0].plot(range(len(new_sig)), new_sig, label='filtered')
        axs[0].vlines(x_axis, ymin=min(self.signal), ymax=max(self.signal), linestyles='dashed')
        axs[0].set_title('Signal')
        axs[0].legend()
        axs[1].stem(range(len(avg)), avg)
        axs[1].set_title('Mean')
        axs[2].stem(range(len(std)), std)
        axs[2].set_title('Std')
        axs[3].stem(range(len(conv)), conv)
        axs[3].set_title('Convolution')
        plt.show()

    def spectrogram(self, seg_len=256, overlap_len=128):

        self.signal = self.filters(1000, 4500, filter_type='bandpass')

        f, t, Sxx = spectrogram(self.signal, fs=self.fs, window='hamming', nperseg=seg_len, noverlap=overlap_len)
        # Sxx = (frequency, time) --> coloumns represent frequencies at a given time
        Sxx = 10*np.log10(Sxx)

        fig, axs = plt.subplots(1, 1)
        axs.pcolormesh(t, f, Sxx, shading='gouraud')
        axs.set_xlabel('Time [sec]')
        axs.set_ylabel('Frequency [Hz]')
        plt.show()

    def correlate(self, signal, window=300):

        avg = [ ]
        for i in range(0, len(signal) - 2 * window, window):
            corr = np.corrcoef(signal[i:i + window], signal[i + window: i + 2*window])
            avg.append(np.mean(corr))
        return avg

    def rolling_window(self, signal, window=256, shift=128):
    
        shape = signal.shape[:-1] + (int(signal.shape[-1] - window + 1), int(window))
        strides = signal.strides + (signal.strides[-1],)
        return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)[::shift]

    def smoothing_filter(self, signal, window=256, filter=np.hamming):

        w = filter(len(signal))
        return signal * w

    def compute_nfft(self, window=256):

        nfft = 1
        while nfft < window:
            nfft *= 2
        return nfft

    def fft(self, signal, window=256):

        fft = np.fft.rfft(signal)
        spectrum = fft * fft / window
        return spectrum

    def avg_power(self, signal, window=256):

        avg = [ ]
        for i in range(0, len(signal), window):
            sig = signal[i:i + window]
            smoothing = self.smoothing_filter(sig)
            fft = self.fft(smoothing)
            avg.append(np.sum(fft) / len(fft))
        return avg


if __name__ == "__main__":
    rate, signal = wavefile.read('./data.wav')
    split = SplitSentence(signal, win_size=300)
    #split.plot(norm=False, filter_type='smooth')
    split.plot(1000, 4500, norm=False, filter_type='bandpass')
