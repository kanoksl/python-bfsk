import math
import pprint

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import scipy.signal._savitzky_golay as savgov
import scipy.fftpack as fftp
import scipy.io.wavfile as wavfile

# -------------------------------------------------------------------------- #

# Use 16-bit PCM format for WAV samples; values range from -32768 to +32767.
# Use numpy's ndarray with dtype='int16'. The axis should be 1 (horizontal).

# Most WAV-related functions will return both WAV array and sample rate,
# even if the sample rate is given as an argument (for consistency).

# -------------------------------------------------------------------------- #

DEFAULT_SAMPLE_RATE = 44100  # int, Hz
DEFAULT_AMPLITUDE = 10000


# ========================================================================== #
#  Utility functions
# ========================================================================== #

def _describe_wav(wav_samples: np.ndarray, sample_rate: int, title=''):
    """ Print information about the WAV samples to the stdout. """
    if title: print(title)
    arr_axis = int(np.argmax(wav_samples.shape))
    n = wav_samples.shape[arr_axis]
    print('    -> sample rate = {:,d} Hz'.format(sample_rate))
    print('    -> number of samples = {:,d}'.format(n))
    print('    -> duration = {:.02f} sec'.format(n / sample_rate))
    print('    --> numpy data type = {}'.format(wav_samples.dtype))
    print('    --> numpy array axis = {}'.format(arr_axis))


def _plot_wav(plot_arg_dicts):
    # pprint.pprint(plot_arg_dicts)
    count = len(plot_arg_dicts)
    plt.figure()
    for i in range(count):
        plt.subplot(count, 1, i + 1)

        title = plot_arg_dicts[i].get('title', None)
        xvals = plot_arg_dicts[i].get('x', None)
        yvals = plot_arg_dicts[i].get('y', None)
        color = plot_arg_dicts[i].get('color', 'black')
        xlabel = plot_arg_dicts[i].get('xlabel', None)
        ylabel = plot_arg_dicts[i].get('ylabel', None)
        xlim = plot_arg_dicts[i].get('xlim', None)
        ylim = plot_arg_dicts[i].get('ylim', None)
        grid = plot_arg_dicts[i].get('grid', False)

        if title is not None: plt.title(title)
        if xvals is not None:
            plt.plot(xvals, yvals, color=color)
        else:
            plt.plot(yvals, color=color)
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        plt.grid(grid)

    # plt.tight_layout()
    plt.subplots_adjust(hspace=1, left=0.05, right=0.95)
    plt.show()


def _plot_wav_analysis(wav_samples, sample_rate, freqs, bw=None, dft=False):
    plot_dicts = list()
    time_domain = np.arange(len(wav_samples)) / sample_rate
    plot_dicts.append({'title': 'original wav samples', 'color': 'red',
                       'x': time_domain, 'xlabel': 'time (sec)',
                       'y': wav_samples, 'ylabel': 'amplitude'})

    if dft:
        arr_axis = int(np.argmax(wav_samples.shape))
        wav_samples_dft = fftp.rfft(wav_samples, axis=arr_axis)
        n = wav_samples_dft.shape[0]
        freq_domain = sample_rate * fftp.rfftfreq(n)

        plot_dicts.append({'title': 'wav samples dft',
                           'x': freq_domain, 'xlabel': 'frequency',
                           'y': wav_samples_dft, 'ylabel': 'amplitude'})
    if type(bw) == int:
        bwl = [bw] * len(freqs)
    elif bw is None:
        bwl = [10] * len(freqs)
    else:
        bwl = bw
    for i, f in enumerate(freqs):
        wav, _ = bandpass_filter2(wav_samples, sample_rate, f, bwl[i])
        plot_dicts.append({'title': 'bandpass filter, fc={}, bw={}'.format(f, bwl[i]),
                           'x': time_domain, 'xlabel': 'time (sec)',
                           'y': wav, 'ylabel': 'amplitude'})

    _plot_wav(plot_dicts)


# ========================================================================== #
#  Digital signal processing (DSP) functions
# ========================================================================== #

def generate_signal(duration=1, samples=None, amplitude=DEFAULT_AMPLITUDE,
                    frequency=1000, sample_rate: int = DEFAULT_SAMPLE_RATE
                    ) -> (np.ndarray, int):
    """
    Generate an array of 16-bit PCM samples with default sample rate
    (defined by DEFAULT_SAMPLE_RATE = 44100 Hz).
    :param duration: duration in seconds.
    :param samples: number of samples; replace duration if assigned.
    :param amplitude: default is 10,000.
    :param frequency: frequency in hertz.
    :param sample_rate: sample rate in hertz.
    :return: a numpy array of WAV samples and the sample rate.
    """
    if samples is None:
        time_domain = np.arange(sample_rate * duration) / sample_rate
    else:
        time_domain = np.arange(samples) / sample_rate
    samples = amplitude * np.sin(2 * np.pi * frequency * time_domain)
    wav_samples = np.array(samples, dtype=np.int16)  # convert to 16-bit
    return wav_samples, sample_rate


# -------------------------------------------------------------------------- #

def bandpass_filter(wav_samples: np.ndarray, sample_rate: int,
                    lower: int, upper: int, plot=False) -> (np.ndarray, int):
    """
    Perform a low-pass and high-pass filter on the WAV samples.
    :param wav_samples: the audio data.
    :param sample_rate: the sample rate of the wav_samples (hertz).
    :param lower: (hertz) all frequencies below this will be blocked.
    :param upper: (hertz) all frequencies above this will be blocked.
    :param plot: plot graphs using matplotlib.
    :return: a new WAV samples with only frequencies in [lower, upper]
             and the sample rate.
    """
    # auto-detect array axis, 0 = vertical, 1 = horizontal
    arr_axis = int(np.argmax(wav_samples.shape))
    wav_samples_dft = fftp.rfft(wav_samples, axis=arr_axis)
    n = wav_samples_dft.shape[0]

    freq = wav_samples_dft_saved = None
    if plot:
        freq = sample_rate * fftp.rfftfreq(n)
        wav_samples_dft_saved = np.copy(wav_samples_dft)

    factor = n / (sample_rate / 2)
    lower_ft = int(lower * factor)
    upper_ft = int(math.ceil(upper * factor))
    wav_samples_dft[:lower_ft] = 0  # high-pass filter
    wav_samples_dft[upper_ft:] = 0  # low-pass filter

    wav_filtered = fftp.irfft(wav_samples_dft, axis=arr_axis)
    wav_filtered = np.round(wav_filtered).astype('int16')

    if plot:
        plot_dicts = list()
        plot_dicts.append({'title': 'original wav samples', 'color': 'red',
                           'y': wav_samples, 'ylabel': 'amplitude'})
        plot_dicts.append({'title': 'original wav samples dft',
                           'x': freq, 'xlabel': 'frequency',
                           'y': wav_samples_dft_saved, 'ylabel': 'amplitude'})
        plot_dicts.append({'title': 'filtered wav samples dft ({}-{} hz)'.format(lower, upper),
                           'x': freq, 'xlabel': 'frequency',
                           'y': wav_samples_dft, 'ylabel': 'amplitude'})
        plot_dicts.append({'title': 'filtered wav samples',
                           'y': wav_filtered, 'ylabel': 'amplitude'})
        _plot_wav(plot_dicts)

    return wav_filtered, sample_rate


def bandpass_filter2(wav_samples: np.ndarray, sample_rate: int, center: int,
                     bandwidth: int, plot=False) -> (np.ndarray, int):
    """ Calculate lower and upper limits from given center frequency and bandwidth
        and simply call the other bandpass_filter(). """
    abs_lower = int(center - bandwidth / 2)
    abs_upper = int(center + bandwidth / 2)
    return bandpass_filter(wav_samples, sample_rate,
                           lower=abs_lower, upper=abs_upper, plot=plot)


def threshold_filter(wav_samples: np.ndarray, sample_rate: int, threshold_value: int,
                     output_low: int, output_high: int, absolute: bool = True
                     ) -> (np.ndarray, int):
    """ Apply a thresholding filter to the WAV samples. """

    def threshold_func(x):
        return output_low if x < threshold_value else output_high

    def threshold_func_abs(x):
        return output_low if abs(x) < threshold_value else output_high

    t_func = np.vectorize(threshold_func_abs if absolute else threshold_func)
    wav_threshold = t_func(wav_samples)

    return wav_threshold, sample_rate


# -------------------------------------------------------------------------- #

class ModulationException(Exception):
    """ Exception occurred in modulation/demodulation process. """
    pass


class ModulatorBFSK:
    """ Modulate digital data using binary frequency shift keying. """

    def __init__(self, f0, f1, symbol_rate, sample_rate=DEFAULT_SAMPLE_RATE):
        """
        Initialize a modulator with the specified configurations.
        :param f0: frequency for bit 0.
        :param f1: frequency for bit 1.
        :param symbol_rate: modulation rate, symbols per second.
        :param sample_rate: sample rate of the audio transmitter.
        """
        super().__init__()

        self.f0 = f0
        self.f1 = f1
        self.symbol_rate = symbol_rate
        self.sample_rate = sample_rate

        self.sps = int(sample_rate / symbol_rate)
        """ samples per symbol """
        self.ts = 1 / symbol_rate
        """ symbol duration time """

        print('[initialize] BFSK modulator/demodulator:')
        print('    -> f0 = {:,d} Hz'.format(f0))
        print('    -> f1 = {:,d} Hz'.format(f1))
        print('    -> symbol rate = {:,d} baud'.format(symbol_rate))
        print('    -> sample rate = {:,d} Hz'.format(sample_rate))
        print('    --> samples per symbol = {:,d} samples'.format(self.sps))
        print('    --> symbol duration time = {:,.2f} sec'.format(self.ts))
        bit_per_symbol = 1
        print('    --> bit per symbol = {:,d} bits'.format(bit_per_symbol))
        print('    --> max data rate = {:,.2f} bps'.format(symbol_rate / bit_per_symbol))

    def modulate(self, bit_stream):
        """ Digital data (bits) -> WAV samples. """
        wav_samples = list()
        bit0_wav, _ = generate_signal(samples=self.sps, frequency=self.f0,
                                      sample_rate=self.sample_rate)
        bit1_wav, _ = generate_signal(samples=self.sps, frequency=self.f1,
                                      sample_rate=self.sample_rate)
        bit0_wav = bit0_wav.tolist()
        bit1_wav = bit1_wav.tolist()

        for bit in bit_stream:
            bit_wav = bit0_wav if bit == BIT0 else bit1_wav
            wav_samples.extend(bit_wav)

        # convert from python list to numpy array
        wav_samples = np.array(wav_samples, dtype=np.int16)

        return wav_samples, self.sample_rate

    def demodulate2(self, wav_samples, sample_rate):
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

        def _smooth(samples, size, threshold, set0=False):
            i = 0
            while i < len(samples) - size:
                buffer = samples[i:i + size]
                if np.sum(buffer) > threshold:
                    for j in range(i, i + size): samples[j] = 1
                elif set0:
                    for j in range(i, i + size): samples[j] = 0
                i += size

        def _average(array, size):
            average = list()
            for i in range(len(array) - size):
                buffer = array[i:i+size]
                average.append(np.average(buffer))
            return average

        def _smooth2(samples, size, threshold):
            result = list()
            for i in range(len(samples) - size):
                buffer = samples[i:i + size]
                if np.sum(buffer) > threshold:
                    result.append(1)
                else:
                    result.append(0)
            return np.array(result)

        wav0 = _bandpass(wav_samples, self.f0)
        wav1 = _bandpass(wav_samples, self.f1)

        wav0 = savgov.savgol_filter(wav0, 99, 3)
        wav1 = savgov.savgol_filter(wav1, 99, 3)

        wav0 = _bthreshold(wav0, t=np.max(wav0) / 3)
        wav1 = _bthreshold(wav1, t=np.max(wav1) / 3)

        wav0 = savgov.savgol_filter(wav0, 55, 2)
        wav1 = savgov.savgol_filter(wav1, 55, 2)

        temp = wav0 * wav1
        print('overlapped: {} samples'.format(np.sum(temp)))

        _plot_wav([
            # {'y': wav_samples, 'title': 'original samples'},
                   {'y': wav0, 'ylim': [0, 1.5], 'title': 'wav0'},
                   {'y': wav1, 'ylim': [0, 1.5], 'title': 'wav1'}])

        # smt = self.sps // 2
        # wav0 = _smooth2(wav0, smt, smt // 3)
        # wav1 = _smooth2(wav1, smt, smt // 3)

        # smt = min(self.sps // 3, 100)
        # print('smt = {}'.format(smt))
        # _smooth(wav0, smt, smt // 3)
        # _smooth(wav1, smt, smt // 3)
        # _smooth(wav0, 3, 2, set0=True)
        # _smooth(wav1, 3, 2, set0=True)
        # _smooth(wav0, 5, 2, set0=True)
        # _smooth(wav1, 5, 2, set0=True)

        temp = wav0 * wav1
        print('smoothed; overlapped: {} samples'.format(np.sum(temp)))

        # trim
        i = 0
        while i < len(wav0) and wav0[i] == 0 == wav1[i]:
            i += 1
        wav0 = wav0[i:]
        wav1 = wav1[i:]
        print('trimmed {} samples from the start'.format(i))

        i = len(wav0) - 1
        while i > 0 and wav0[i] == 0 == wav1[i]:
            i -= 1
        wav0 = wav0[:i + 1]
        wav1 = wav1[:i + 1]
        print('trimmed {} samples from the end'.format(i))

        l0 = []
        for i in range(self.sps):
            j = i
            val, cnt = 0, 0
            while j < len(wav0):
                if wav0[j] != wav1[j]:
                    val += 1
                cnt += 1
                j += self.sps
            l0.append(val / cnt)

        temp = l0.copy()
        temp.extend(l0[:self.sps // 2])
        l0 = _average(temp, self.sps // 2)


        #
        # print(l0)
        # ss = l0[-1][1]

        # l0.extend(l0)
        #
        # plt.figure()
        # plt.plot(l0)
        # plt.show()
        #
        # ssl = []
        # ss = 0
        # cf_interval = int(0.7 * self.sps)
        # for i in range(self.sps):
        #     buf = l0[i:i + cf_interval]
        #     if np.sum(buf) == cf_interval:
        #         ss = i + cf_interval // 2
        #         ssl.append(ss)
        #
        # print(ssl)

        # ssl2 = ssl.copy()
        # ssl2.append(ssl2.pop(0))
        # diff = [abs(ssl[k] - ssl2[k]) for k in range(len(ssl))]
        # diff = diff[10:-10]
        # max = np.max(diff)
        # print('ssl-ssl2 max: {}'.format(max))
        #
        # if max < 5:
        #     ss = ssl[len(ssl) // 2]
        # else:
        #     a = int(np.argmax(diff))
        #     idx = (a + len(ssl) // 2) % len(ssl)
        #     ss = ssl[len(idx)]

        # ss_l = len(l0) - 1
        # ss_r = 0
        # stop_l = stop_r = False
        # for i in range(len(l0)):
        #     if l0[i] >= l0[ss_l] and i < ss_l:
        #         ss_l = i
        #     if l0[i] >= l0[ss_r] and i > ss_r:
        #         ss_r = i
        #
        # l0[ss_l] = 2
        # l0[ss_r] = 2
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.plot(temp)
        # plt.subplot(2, 1, 2)
        # plt.plot(l0)
        # plt.show()
        #
        # print('ss_l, ss_r = {}, {}'.format(ss_l, ss_r))
        #
        # if ss_l == 0 and ss_r == len(l0) - 1:
        #     ss = 0
        # else:
        #     ss = (ss_l + ss_r) // 2

        ss = int(self.sps * 4 / 7)

        # wav0[ss] = 2
        _plot_wav([
            # {'y': wav_samples, 'title': 'original samples'},
                   {'y': wav0, 'ylim': [0, 1.5], 'title': 'wav0 trimmed, smoothed'},
                   {'y': wav1, 'ylim': [0, 1.5], 'title': 'wav1 trimmed, smoothed'}])

        print('ss = {}'.format(ss))
        # ss0 = np.argmin(l0)
        # ss1 = np.argmax(l1)
        # if abs(ss0 - ss1) < self.sps / 2:
        #     ss = (ss0 + ss1) // 2
        #     print('  ss ok')

        sc = int((len(wav_samples) - ss) // self.sps)
        se = int(sc * self.sps)
        print('len = {}, se = {}, sc = {}'.format(len(wav_samples), se, sc))
        # wav_samples = wav_samples[ss:se + 1]
        wav0 = wav0[ss:se + 1]
        wav1 = wav1[ss:se + 1]

        # print('ss0 = {}, ss1 = {}'.format(ss0, ss1))

        i = 0
        while i < len(wav0):
            d = self.sps // 6
            cnt0 = 0
            cnt1 = 0
            for j in range(i-d, i+d+1):
                if j < 0 or j >= len(wav0): break
                if wav0[j] > wav1[j]:
                    cnt0 += 1
                else:
                    cnt1 += 1
            # if wav0[i] > wav1[i]:
            #     bit_stream.append(BIT0)
            # else:
            #     bit_stream.append(BIT1)

            bit_stream.append(BIT0 if cnt0 > cnt1 else BIT1)

            wav0[i] = 2
            wav1[i] = 2
            i += self.sps

        _plot_wav([
            # {'y': wav_samples, 'title': 'original samples'},
                   {'y': wav0, 'ylim': [0, 1.5], 'title': 'wav0 sampling'},
                   {'y': wav1, 'ylim': [0, 1.5], 'title': 'wav1 sampling'}])

        return bit_stream

    def demodulate(self, wav_samples, sample_rate):
        """ WAV samples -> original data bits. """
        bit_stream = list()
        symbol_count = int(len(wav_samples) / self.sps)

        def bandpass(samples, f):
            result, _ = bandpass_filter2(samples, sample_rate,
                                         center=f, bandwidth=100,
                                         plot=False)
            return result

        def func(x):
            return x * x

        func = np.vectorize(func)

        # locate the start of transmission stream
        buffer_len = 2 * FLAG_LEN
        flag_wav, _ = self.modulate(FLAG_SEQUENCE)
        flag_wav = bandpass(flag_wav, self.f0) + bandpass(flag_wav, self.f1)
        flag_wav = abs(flag_wav.astype(np.float))
        print(np.max(flag_wav))
        flag_wav /= np.max(flag_wav)
        # print('flagwav = ' + str(flag_wav))

        sim_pos = list()
        sim_val = list()
        flag_pos = list()

        for i in range(symbol_count // buffer_len * 2):
            slice_idx = i * FLAG_LEN * self.sps
            buffer = wav_samples[slice_idx:slice_idx + buffer_len * self.sps]
            for j in range(buffer_len - FLAG_LEN):
                if (j + FLAG_LEN) * self.sps > len(buffer): break
                buffer2 = buffer[j * self.sps:(j + FLAG_LEN) * self.sps]
                # print(buffer2)
                buffer2 = bandpass(buffer2, self.f0) + bandpass(buffer2, self.f1)
                # print(buffer2)
                # print(np.max(buffer2))
                buffer2 = abs(buffer2.astype(np.float))
                if np.max(buffer2) != 0:
                    buffer2 /= np.max(buffer2)  # normalize
                    # similarity = np.dot(flag_wav, buffer2) / np.dot(flag_wav, flag_wav) * 100
                    buffer2 -= flag_wav
                    buffer2 = func(buffer2)
                    similarity = -1 * np.sum(buffer2)
                else:
                    similarity = 0
                print('similarity @ {} = {:.4f}'.format(slice_idx + j * self.sps, similarity))
                # print('  ' + str(flag_wav))
                # print('  ' + str(buffer2))
                # if similarity > 95:  # found flag sequence
                #     flag_pos.append(slice_idx + j * self.sps)
                sim_pos.append(slice_idx + j * self.sps)
                sim_val.append(similarity)

        plt.figure()
        plt.plot(sim_pos, sim_val)
        plt.show()

        print('found {} flags: at {}'.format(len(flag_pos), flag_pos))
        if len(flag_pos) < 2:
            print('cannot find flag with 95% confidence; using relative approach...')

            i0 = np.argmax(sim_val)
            flag_pos.append(sim_pos[i0])
            sim_val[i0] = np.min(sim_val)
            i0 = np.argmax(sim_val)
            flag_pos.append(sim_pos[i0])
            print('found {} flags: at {}'.format(len(flag_pos), flag_pos))

            # raise RuntimeError('too few flag sequences')
        elif len(flag_pos) > 2:
            raise RuntimeError('too many flag sequences')

        print(len(wav_samples))
        wav_samples = wav_samples[flag_pos[0]:flag_pos[1] + FLAG_LEN * self.sps]

        # demodulate the samples
        symbol_count = int(len(wav_samples) / self.sps)
        for i in range(symbol_count):
            slice_idx = i * self.sps
            symbol_samples = wav_samples[slice_idx:slice_idx + self.sps]
            f0_samples = bandpass(symbol_samples, self.f0)
            f1_samples = bandpass(symbol_samples, self.f1)
            val0 = np.sum(abs(f0_samples))
            val1 = np.sum(abs(f1_samples))
            bit_stream.append(BIT0 if val0 > val1 else BIT1)

        return bit_stream


# ========================================================================== #
#  Digital encoding functions
# ========================================================================== #

# constants for bit values
BIT0 = '0'
BIT1 = '1'

# maximum number of consecutive 1s allowed in data bits
STUFFING_LEN = 29

# signals the beginning/ending of the data stream
# FLAG_SEQUENCE = [BIT0] + ([BIT1] * (STUFFING_LEN + 1)) + [BIT0]
FLAG_SEQUENCE = ([BIT1] * (STUFFING_LEN + 1))
FLAG_LEN = len(FLAG_SEQUENCE)


def get_transmission_bit_stream(data_bit_stream):
    """ Encode the data bits with additions such as adding start sequence
        or bit stuffing. The result bits are ready to be transmitted.
    """
    t_stream = list(data_bit_stream)

    # perform bit stuffing; don't allow too many consecutive 1s in a row
    i = j = 0
    while i < len(t_stream):
        for j in range(i, i + STUFFING_LEN + 1):
            if j == len(t_stream) or t_stream[j] == BIT0:
                break
        else:  # no break; all 1s
            t_stream.insert(j, BIT0)
        i = j + 1

    # prepend/append flag sequences
    temp = list(FLAG_SEQUENCE)
    temp.extend(t_stream)
    t_stream = temp
    t_stream.extend(FLAG_SEQUENCE)

    # print(t_stream)
    return t_stream


def get_data_bit_stream(transmission_bit_stream):
    """ Reverse of the get_transmission_bit_stream(). """
    d_stream = list(transmission_bit_stream)

    if not (d_stream[:FLAG_LEN] == FLAG_SEQUENCE == d_stream[-FLAG_LEN:]):
        print('error: missing start/end flag sequences')
        print(''.join(d_stream))
        # raise RuntimeError('bit stream does not have begin/end flag sequences')

    # remove flag sequences
    d_stream = d_stream[FLAG_LEN:-FLAG_LEN]

    # un-stuffing; remove stuffed 0 after every some number of consecutive 1s
    i = j = 0
    while i < len(d_stream):
        for j in range(i, i + STUFFING_LEN):
            if j == len(d_stream) or d_stream[j] == BIT0:
                break
        else:  # no break; all 1s
            if d_stream[j + 1] == BIT0:
                d_stream.pop(j + 1)
            else:
                print('error: incorrect bit stuffing pattern at ' + str(j))
                # raise RuntimeError('error in bit stuffing pattern')
        i = j + 1

    # print(d_stream)
    return d_stream


# ========================================================================== #
#  Input/output (IO) functions
# ========================================================================== #

def input_from_microphone(duration, sample_rate: int = DEFAULT_SAMPLE_RATE
                          ) -> (np.ndarray, int):
    """ Microphone -> WAV samples. """
    print('[begin] input_from_microphone: {} sec'.format(duration))
    sample_count = int(duration * sample_rate)
    wav_samples = sd.rec(sample_count, samplerate=sample_rate,
                         channels=1, dtype=np.int16)
    sd.wait()

    # convert from vertical array to horizontal (normal) array
    wav_samples = wav_samples.transpose()[0]

    print('[done] input_from_microphone: {} sec'.format(duration))
    _describe_wav(wav_samples, sample_rate)

    return wav_samples, sample_rate


def input_from_file(file_path: str) -> (np.ndarray, int):
    """ Audio file (.wav) -> WAV samples. """
    sample_rate, wav_samples = wavfile.read(file_path)

    print('[done] input_from_file: ' + file_path)
    _describe_wav(wav_samples, sample_rate)

    return wav_samples, sample_rate


def output_to_speaker(wav_samples: np.ndarray, sample_rate: int):
    """ WAV samples -> speaker. """
    sd.play(wav_samples, samplerate=sample_rate, blocking=True)

    print('[done] output_to_speaker')
    _describe_wav(wav_samples, sample_rate)


def output_to_file(wav_samples: np.ndarray, sample_rate: int, file_path: str,
                   padding_duration=0):
    """ WAV samples -> audio file (.wav). """
    if padding_duration > 0:
        padded = [0] * int(sample_rate * padding_duration)
        padded.extend(wav_samples.tolist())
        padded.extend([0] * int(sample_rate * padding_duration))
        wav_samples = np.array(padded, dtype=np.int16)

    wavfile.write(file_path, sample_rate, wav_samples)

    print('[done] output_to_file' + file_path)
    _describe_wav(wav_samples, sample_rate)


# -------------------------------------------------------------------------- #

def read_file(file_path: str):
    """ Read a file and convert it to bit string, ready to be modulated. """
    pass


def write_file(bit_string, file_path: str):
    """ Convert a bit string to binary data and write to a file. """
    pass


# ========================================================================== #
#  Tester
# ========================================================================== #


def run_testcase(f, duration=0):
    f.readline()
    f0 = int(f.readline().strip())
    f1 = int(f.readline().strip())
    br = int(f.readline().strip())
    data = f.readline().strip()

    modem = ModulatorBFSK(f0, f1, symbol_rate=br)

    if not duration:
        duration = len(data) / br * 2

    w, sr = input_from_microphone(duration)
    # w, sr = input_from_file('{}.wav'.format(f.name))

    b = modem.demodulate2(w, sr)
    b = get_data_bit_stream(b)

    print('data: ' + data)
    print('recv: ' + ''.join(b))

    error = 0
    print('diff: ', end='')
    for i in range(min(len(b), len(data))):
        if str(b[i]) != str(data[i]):
            print('x', end='')
            error += 1
        else:
            print('_', end='')
    print()
    print('len = {} bits, received = {} bits'.format(len(data), len(b)))
    print('error: {} bits ({:.2f} %)'.format(error, error / len(data) * 100))


def gen_testcase(length, f0, f1, br, filename):
    import random
    with open('testcases/' + filename, 'w') as f:
        f.write('bfsk\n')
        f.write('{}\n{}\n{}\n'.format(f0, f1, br))
        data = []
        for i in range(length):
            if random.random() < 0.5:
                data.append(BIT0)
            else:
                data.append(BIT1)
        f.write(''.join(data))

    modem = ModulatorBFSK(f0, f1, symbol_rate=br)
    t = get_transmission_bit_stream(data)
    # print(t)
    w, sr = modem.modulate(t)
    output_to_file(w, sr, '{}.wav'.format(f.name), padding_duration=1)


if __name__ == '__main__':
    # gen_testcase(2000, 7000, 10000, 100, '01L.txt')

    with open('testcases/01L.txt') as f:
        run_testcase(f, duration=30)

    # w, sr = input_from_microphone(30)
    # w, sr = input_from_file('wav500hz.wav')
    # w, sr = bandpass_filter2(w, sr, center=700, bandwidth=10, plot=True)
    # output_to_speaker(w, sr)

    # freq = 10000
    # wave, sr = generate_signal(60, frequency=freq)
    # output_to_file(wave, sr, 'wav{}hz.wav'.format(freq))

    # with open('randombits.txt') as f:
    #     d0 = f.read(1000)
    #     print(d0)
    #
    #     print('data stream length = ' + str(len(d0)))
    #     t = get_transmission_bit_stream(d0)
    #     print('transmission stream length = ' + str(len(t)))
    #
    #     print(''.join(t))
    #
    #     modem = ModulatorBFSK(f0=500, f1=1000, symbol_rate=100)
    #     w, sr = modem.modulate(t)
    #
    #     # output_to_speaker(w, sr)
    #     # output_to_file(w, sr, 'random.wav')
    #
    #     b = modem.demodulate(w, sr)
    #
    #     print(''.join(b))
    #     print(''.join(t))
    #
    #     print(b == t)

    # modem = ModulatorBFSK(f0=7000, f1=10000, symbol_rate=10)
    # w, sr = input_from_microphone(20)
    #
    # _plot_wav_analysis(w, sr, [7000, 10000], bw=100, dft=True)
    #
    # # modem.demodulate2(w, sr)
    #
    # # bandpass_filter2(w, sr, 7000, 100, True)
    # # bandpass_filter2(w, sr, 10000, 100, True)
    # #
    # b = modem.demodulate2(w, sr)
    # b = get_data_bit_stream(b)
    # with open('randombits.txt') as f:
    #     d0 = f.read(100)
    #
    #     # t = get_transmission_bit_stream(d0)
    #     # w, sr = modem.modulate(t)
    #     # output_to_file(w, sr, 'random4_hf_seq32_7000_10000_br10.wav')
    #
    #     print(d0)
    #     print(''.join(b))
    #
    #     print('diff')
    #     for i in range(min(len(b), len(d0))):
    #         if str(b[i]) != str(d0[i]):
    #             print('x', end='')
    #         else:
    #             print('_', end='')



