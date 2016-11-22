
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftp

import audio_processing as ap

import util
def print_b(string): util.print_b(util.CYAN, string)
def print_c(string): util.print_c(util.CYAN, string)

# -------------------------------------------------------------------------- #


def describe_wav(wav_samples: np.ndarray, sample_rate: int, title=''):
    """ Print information about the WAV samples to the stdout. """
    if not title:
        title = 'WAV samples description:'
    arr_axis = int(np.argmax(wav_samples.shape))
    n = wav_samples.shape[arr_axis]
    print_b(title)
    print_c(' - sample rate = {:,d} Hz'.format(sample_rate))
    print_c(' - number of samples = {:,d}'.format(n))
    print_c(' - duration = {:.02f} sec'.format(n / sample_rate))
    # print_c(' - numpy data type = {}'.format(wav_samples.dtype))
    # print_c(' - numpy array axis = {}'.format(arr_axis))


def plot_wav(plot_arg_dicts):
    """ Plot and show the graphs specified in the argument. """
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

    plt.subplots_adjust(hspace=1, left=0.05, right=0.95)
    plt.show()


def plot_wav_analysis(wav_samples, sample_rate, freqs=[], bw=None, dft=False):
    """ Perform DFT on the samples, select the given frequencies, and plot."""
    plot_dicts = list()
    time_domain = np.arange(len(wav_samples)) / sample_rate
    plot_dicts.append({'title': 'wav samples', 'color': 'red',
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
        wav, _ = ap.bandpass_filter(wav_samples, sample_rate, f, bwl[i])
        plot_dicts.append({'title': 'bandpass filter, fc={}, bw={}'.format(f, bwl[i]),
                           'x': time_domain, 'xlabel': 'time (sec)',
                           'y': wav, 'ylabel': 'amplitude'})
    plot_wav(plot_dicts)
