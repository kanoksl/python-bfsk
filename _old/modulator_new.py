from _old.audio_all_clean import *


class ModulatorMFSK:
    """ Modulate digital data using multi-level frequency shift keying. """

    def __init__(self, sym_freq_dict, symbol_rate, sample_rate=DEFAULT_SAMPLE_RATE):
        """
        Initialize a modulator with the specified configurations.
        :param sym_freq_dict: frequency for each symbol.
        :param symbol_rate: modulation rate, symbols per second.
        :param sample_rate: sample rate of the audio transmitter.
        """
        super().__init__()

        self.sfdict = sym_freq_dict
        self.symbol_rate = symbol_rate
        self.sample_rate = sample_rate

        self.sps = int(sample_rate / symbol_rate)
        """ samples per symbol """
        self.ts = 1 / symbol_rate
        """ symbol duration time """

        print('[initialize] BFSK modulator/demodulator:')
        for symbol in sorted(sym_freq_dict.keys()):
            print('    -> f_{} = {:,d} Hz'.format(symbol, sym_freq_dict[symbol]))
        print('    -> symbol rate = {:,d} baud'.format(symbol_rate))
        print('    -> sample rate = {:,d} Hz'.format(sample_rate))
        print('    --> samples per symbol = {:,d} samples'.format(self.sps))
        print('    --> symbol duration time = {:,.2f} sec'.format(self.ts))
        self.bit_per_symbol = len(sym_freq_dict.keys()[0])
        print('    --> bit per symbol = {:,d} bits'.format(self.bit_per_symbol))
        print('    --> max data rate = {:,.2f} bps'.format(symbol_rate / self.bit_per_symbol))

    def to_symbol_stream(self, bit_stream):
        symbol_stream = list()
        for i in range(0, len(bit_stream), self.bit_per_symbol):
            symbol_stream.append(''.join(bit_stream[i:i+self.bit_per_symbol]))
        return symbol_stream

    def modulate(self, bit_stream):
        """ Digital data (bits) -> WAV samples. """
        wav_samples = list()
        mdwavs = dict()
        for symbol in sorted(sym_freq_dict.keys()):
            wav, _ = generate_signal(samples=self.sps, frequency=self.sfdict[symbol],
                                     sample_rate=self.sample_rate)
            mdwavs[symbol] = wav.tolist()

        for symbol in self.to_symbol_stream(bit_stream):
            wav_samples.extend(mdwavs[symbol])

        # convert from python list to numpy array
        wav_samples = np.array(wav_samples, dtype=np.int16)

        return wav_samples, self.sample_rate

    def demodulate(self, wav_samples, sample_rate, smoothing=True):
        """ WAV samples -> original data bits. """
        print('[begin] demodulation process')
        bit_stream = list()

        def _bandpass(samples, f):
            result, _ = bandpass_filter2(samples, sample_rate, center=f, bandwidth=100)
            return result

        def _bthreshold(samples, t=10):
            print('_bthreshold, t={}'.format(t))
            result, _ = threshold_filter(samples, sample_rate, t, 0, 1)
            return result

        def _average(array, size):
            average = list()
            for i in range(len(array) - size):
                buffer = array[i:i + size]
                average.append(np.average(buffer))
            return average

        wavs = dict()
        for symbol in sorted(self.sfdict.keys()):
            w = _bandpass(wav_samples, self.sfdict[symbol])
            if smoothing:
                w = savgov.savgol_filter(w, 99, 3)
            w = _bthreshold(w, t=np.max(w) / 3)
            if smoothing:
                w = savgov.savgol_filter(w, 55, 2)
            wavs[symbol] = w

        pltdicts = list()
        for symbol in sorted(self.sfdict.keys()):
            pltdicts.append({'y': wavs[symbol], 'ylim': [0, 1.5], 'title': 'wav_' + symbol})
        _plot_wav(pltdicts)

        # trim start
        i = 0
        while i < len(wav0) and wav0[i] == 0 == wav1[i]:
            i += 1
        wav0 = wav0[i:]
        wav1 = wav1[i:]
        print('trimmed {} samples from the start'.format(i))

        # trim end
        i = len(wav0) - 1
        while i > 0 and wav0[i] == 0 == wav1[i]:
            i -= 1
        wav0 = wav0[:i + 1]
        wav1 = wav1[:i + 1]
        print('trimmed {} samples from the end'.format(i))

        # approximately the middle of the first bit
        ss = int(self.sps * 4 / 7)

        _plot_wav([
            # {'y': wav_samples, 'title': 'original samples'},
            {'y': wav0, 'ylim': [0, 1.5], 'title': 'wav0 trimmed, smoothed'},
            {'y': wav1, 'ylim': [0, 1.5], 'title': 'wav1 trimmed, smoothed'}])

        sc = int((len(wav_samples) - ss) // self.sps)  # bit count
        se = int(sc * self.sps)  # position of the last bit

        print('ss = {}'.format(ss))
        print('len = {}, se = {}, sc = {}'.format(len(wav_samples), se, sc))

        wav0 = wav0[ss:se + 1]
        wav1 = wav1[ss:se + 1]

        i = 0
        d = self.sps // 6  # width of the samples in a bit to compare
        while i < len(wav0):
            cnt0 = 0  # number of times wav0 is stronger than wav1
            for j in range(max(0, i - d), i + d + 1):
                if j < 0 or j >= len(wav0): break
                cnt0 += 1 if wav0[j] > wav1[j] else (-1)
            # print('@bitpos {} (chkrange {} to {}): cnt0 = {}'.format(i, i-d, i+d+1, cnt0))
            bit_stream.append(BIT0 if cnt0 >= 0 else BIT1)

            # mark bit for plotting
            wav0[i] = 2
            wav1[i] = 2
            i += self.sps

        _plot_wav([
            # {'y': wav_samples, 'title': 'original samples'},
            {'y': wav0, 'ylim': [0, 1.5], 'title': 'wav0 sampling'},
            {'y': wav1, 'ylim': [0, 1.5], 'title': 'wav1 sampling'}])

        return bit_stream