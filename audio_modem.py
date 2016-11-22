
import numpy as np
import scipy.signal as signal
import scipy.signal.signaltools as sigtool

import audio_processing as ap
import configuration as cf
import plotter as pt
from digital_data import BIT0, BIT1

import util
def print_b(string): util.print_b(util.BLUE, string)
def print_c(string): util.print_c(util.BLUE, string)

# -------------------------------------------------------------------------- #


class ModulatorDemodulator:
    """ Modulate/demodulate digital data using binary frequency-shift-keying (BFSK). """

    def __init__(self, config, sample_rate=cf.DEFAULT_SAMPLE_RATE):
        """
        Initialize a modulator with the specified configurations.
        :param config: the modulator configuration (bit rate, f0, f1).
        :param sample_rate: sample rate of the audio transmitter.
        """
        super().__init__()

        # basic properties
        self.f0 = config.f0
        """ frequency for bit 0 (hertz). """
        self.f1 = config.f1
        """ frequency for bit 1 (hertz). """
        self.mr = config.bit_rate
        """ modulation rate, or symbol rate (baud), equals to bit rate (bps) in this case. """
        self.sr = sample_rate
        """ audio sample rate (hertz). """

        # calculated properties
        self.sps = int(sample_rate / self.mr)
        """ samples per symbol = [sample rate / symbol rate]. """
        self.ts = 1 / self.mr
        """ symbol duration time, or time per symbol (seconds) = [1 / symbol rate]. """
        self.bps = 1  # not used
        """ bits per symbol (bits). """

        print_c('Initialized: {}'.format(self))

    def __str__(self):
        description = 'BFSK Modulator/Demodulator:'
        description += '\n - f0 = {:,d} Hz'.format(self.f0)
        description += '\n - f1 = {:,d} Hz'.format(self.f1)
        description += '\n - symbol rate = {:,d} baud'.format(self.mr)
        description += '\n - sample rate = {:,d} Hz'.format(self.sr)
        description += '\n - samples per symbol = {:,d} samples'.format(self.sps)
        description += '\n - symbol duration time = {:,.4f} sec'.format(self.ts)
        description += '\n - bit per symbol = {:,d} bits'.format(self.bps)
        description += '\n - max data rate = {:,.2f} bps'.format(self.mr / self.bps)
        return description

    def modulate(self, bits):
        """
        Modulate the digital bits into WAV audio data. Using the initialized configuration.

        :param bits: list of '0's and '1's.
        :return: a tuple (wav_samples, sample_rate).
        """
        duration = float(len(bits)) / float(self.mr)

        t = np.arange(0, duration, 1 / float(self.sr), dtype=np.float)
        m = np.zeros(0).astype(float)
        for bit in bits:
            m = np.hstack((m, np.multiply(np.ones(self.sps), self.f0 if bit == BIT0 else self.f1)))

        t = t[:len(m)]

        wav_samples = 10000 * np.cos(2 * np.pi * np.multiply(m, t))
        wav_samples = wav_samples.astype(np.int16)

        print_b('modulation: finished modulating {:,d} bits'.format(len(bits)))
        return wav_samples, self.sr

    def demodulate(self, wav_samples, sample_rate):
        """
        Demodulate the audio into digital data bits.

        :param wav_samples: the audio data.
        :param sample_rate: the sample rate of wav_samples (hertz).
        :return: list of '0's and '1's.
        """
        bits = list()

        # step 1: perform bandpass filter, keeping only frequencies around f0 and f1
        if False:
            print_b('demodulation: performing bandpass filtering...')
            bandpass_width = 1000
            print_c('  using bandwidth = {:,d} hz'.format(bandpass_width))
            wav_f0, _ = ap.bandpass_filter(wav_samples, sample_rate, self.f0, bandpass_width)
            wav_f1, _ = ap.bandpass_filter(wav_samples, sample_rate, self.f1, bandpass_width)
            wav_samples = wav_f0 + wav_f1

        # step 2: trim blank audio at start/end of recording
        print_b('demodulation: trimming start/end...')
        mean = np.mean(wav_samples)
        sdev = np.std(wav_samples)

        print_c('  wav_samples mean = {}'.format(mean))
        print_c('  wav_samples sdev = {}'.format(sdev))

        threshold = sdev  # amplitude below this will be counted as silence
        i = 0  # start index
        j = len(wav_samples) - 1  # end index
        while abs(wav_samples[i]) < threshold: i += 1
        while abs(wav_samples[j]) < threshold: j -= 1
        wav_samples = wav_samples[i:j + 2]

        n = len(wav_samples)
        print_c('  trim start: {:,d} (at {:.2f} sec)'.format(i, i / sample_rate))
        print_c('  trim end:   {:,d} (at {:.2f} sec)'.format(j, j / sample_rate))
        print_c('  wav data length: {:,d} samples ({:.2f} sec)'.format(n, n / sample_rate))
        print_c('      = bit count: {:,d} bits'.format(n // self.sps))

        pt.plot_wav_analysis(wav_samples, sample_rate)

        # step 3: demodulate using frequency detection algorithm defined in audio_processing.py
        print_b('demodulation: detecting frequency and converting to bits...')
        step = True
        samples_per_symbol = self.sps
        margin = int(0.05 * samples_per_symbol)
        i = 0
        while i + samples_per_symbol < len(wav_samples):
            p = i + margin  # the start index of this symbol
            i += samples_per_symbol  # move to next symbol
            q = i - margin  # the end index of this symbol
            symbol_wav = wav_samples[p:q]

            freq = ap.detect_frequency_2(symbol_wav, sample_rate)

            if step:
                print('  analyzing symbol samples: {}'.format(symbol_wav))
                print('    detected f = {}'.format(freq))
                if input():
                    step = False
                    print('  skip to the end')

            d0 = abs(freq - self.f0)
            d1 = abs(freq - self.f1)
            bits.append(BIT0 if d0 < d1 else BIT1)

        print_b('demodulation: finished demodulating {:,d} bits'.format(len(bits)))
        return bits
