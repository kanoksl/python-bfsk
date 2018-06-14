import numpy as np
import scipy.fftpack as fftp

import configuration as cf
import plotter as pt
import util

# -------------------------------------------------------------------------- #


def generate_signal(duration=1, sample_count=None, frequency=1000,
                    amplitude=cf.DEFAULT_AMPLITUDE, sample_rate=cf.DEFAULT_SAMPLE_RATE):
    """
    Generate an array of 16-bit PCM samples.

    :param duration: duration in seconds.
    :param sample_count: number of samples; replace duration if assigned.
    :param frequency: frequency in hertz.
    :param amplitude: default is 10,000.
    :param sample_rate: sample rate in hertz, default is 44,100 Hz.
    :return: a numpy array of WAV samples and the sample rate.
    """
    if sample_count is None:
        sample_count = sample_rate * duration
    time_domain = np.arange(sample_count) / sample_rate
    sample_count = amplitude * np.sin(2 * np.pi * frequency * time_domain)
    wav_samples = np.array(sample_count, dtype=np.int16)  # convert to 16-bit
    return wav_samples, sample_rate

# -------------------------------------------------------------------------- #


def bandpass_filter0(wav_samples, sample_rate, lower, upper, plot=False):
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
    upper_ft = int(np.ceil(upper * factor))
    wav_samples_dft[:lower_ft] = 0  # high-pass filter
    wav_samples_dft[upper_ft:] = 0  # low-pass filter

    wav_filtered = fftp.irfft(wav_samples_dft, axis=arr_axis)
    wav_filtered = np.round(wav_filtered).astype('int16')

    if plot:  # plot the results
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
        pt.plot_wav(plot_dicts)

    return wav_filtered, sample_rate


def bandpass_filter(wav_samples, sample_rate, center, bandwidth, plot=False):
    """
    Perform a low-pass and high-pass filter on the WAV samples.

    :param wav_samples: the audio data.
    :param sample_rate: the sample rate of the wav_samples (hertz).
    :param center: (hertz) the center of the frequency to keep.
    :param bandwidth: (hertz) the width of the frequency part to keep.
    :param plot: plot graphs using matplotlib.
    :return: a new WAV samples with only frequencies in [center-bw/2, center+bw/2]
             and the sample rate.
    """
    abs_lower = int(center - bandwidth / 2)
    abs_upper = int(center + bandwidth / 2)
    return bandpass_filter0(wav_samples, sample_rate, abs_lower, abs_upper, plot=plot)


def threshold_filter(wav_samples, threshold_value, output_low, output_high, absolute=True):
    """
    Apply a thresholding filter to the WAV samples.

    :param wav_samples: the audio data.
    :param threshold_value: the thresholding point.
    :param output_low: data points below the t-value are changed to this.
    :param output_high: data points above the t-value are changed to this.
    :param absolute: check the absolute value of data instead of the data as-is.
    :return: a new WAV samples that is the result of thresholding.
    """

    def threshold_func(x):
        return output_low if x < threshold_value else output_high

    def threshold_func_abs(x):
        return output_low if abs(x) < threshold_value else output_high

    t_func = np.vectorize(threshold_func_abs if absolute else threshold_func)
    wav_threshold = t_func(wav_samples)

    return wav_threshold


def detect_frequency_1(wav_samples, sample_rate):
    """
    Approximating the frequency of the audio by counting the total number of peaks.

    :param wav_samples: the audio data.
    :param sample_rate: the sample rate of the wav_samples (hertz).
    :return: approximated frequency of the audio (hertz).
    """
    peak_count = 0
    for i in range(2, len(wav_samples)):
        if wav_samples[i] < wav_samples[i - 1] and wav_samples[i - 1] > wav_samples[i - 2]:
            peak_count += 1

    # print(peak_count)

    duration = len(wav_samples) / sample_rate
    freq = peak_count / duration

    # print('freq = {}'.format(freq))
    return freq


def detect_frequency_2(wav_samples, sample_rate):
    """
    Approximating the frequency by finding the time difference between adjacent peaks.

    :param wav_samples: the audio data.
    :param sample_rate: the sample rate of the wav_samples (hertz).
    :return: approximated frequency of the audio (hertz).
    """
    peak_count = 0
    last_peak = 0
    first_peak = -1
    diff_sum = 0
    for i in range(2, len(wav_samples)):
        if wav_samples[i] < wav_samples[i - 1] and wav_samples[i - 1] > wav_samples[i - 2]:
            if first_peak < 0:
                first_peak = i
            peak_count += 1
            diff_sum += i - last_peak
            last_peak = i
    diff_sum -= first_peak

    #    _plot_wav_analysis(wav_samples, sample_rate)

    if peak_count == 1:
        diff_avg = first_peak
    else:
        diff_avg = diff_sum / (peak_count - 1)
    freq = sample_rate / diff_avg

    #    print('d_avg={}, pc={}, freq={}'.format(diff_avg, peak_count, freq))
    #    pprint.pprint(wav_samples)
    #    x = input()
    return freq

    
def detect_frequency_3(wav_samples, sample_rate):
    last_crossing = -1
    for i in range(1, len(wav_samples)):
        wa, wb = wav_samples[i - 1], wav_samples[i]
        
    
    
# -------------------------------------------------------------------------- #


if __name__ == '__main__':
    pass