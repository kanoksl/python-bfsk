import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd

import configuration as cf
import plotter as pt

import util
def print_b(string): util.print_b(util.MAGENTA, string)
def print_c(string): util.print_c(util.MAGENTA, string)

# -------------------------------------------------------------------------- #

def input_from_microphone(duration, sample_rate=cf.DEFAULT_SAMPLE_RATE):
    """
    Record the sound from the system's microphone.

    :param duration: Duration in seconds.
    :param sample_rate: Sample rate. Default is 44100 Hz.
    :return: A tuple, (wav_samples, sample_rate).
    """
    print_b('[begin] input_from_microphone: {} sec'.format(duration))

    sample_count = int(duration * sample_rate)
    wav_samples = sd.rec(sample_count, samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()

    # convert from vertical array to horizontal (normal) array
    wav_samples = wav_samples.transpose()[0]

    print_c('[done] input_from_microphone: {} sec'.format(duration))
    pt.describe_wav(wav_samples, sample_rate)
    return wav_samples, sample_rate


def input_from_file(file_path):
    """
    Read audio data from WAV file.

    :param file_path: Path of the .wav file to read from.
    :return: A tuple, (wav_samples, sample_rate).
    """
    sample_rate, wav_samples = wavfile.read(file_path)

    print_c('[done] input_from_file: ' + file_path)
    pt.describe_wav(wav_samples, sample_rate)
    return wav_samples, sample_rate


def output_to_speaker(wav_samples, sample_rate=cf.DEFAULT_SAMPLE_RATE):
    """
    Play the given WAV data through the system's speaker.

    :param wav_samples: A numpy array of audio amplitudes.
    :param sample_rate: Sample rate. Default is 44100 Hz.
    """
    sd.play(wav_samples, samplerate=sample_rate, blocking=True)

    print_c('[done] output_to_speaker')
    pt.describe_wav(wav_samples, sample_rate)


def output_to_file(file_path, wav_samples, sample_rate, padding_duration=0, csv=False):
    """
    Write the given WAV data to a .wav file (and optionally a .csv file).

    :param file_path: Path of the .wav file to write to.
    :param wav_samples: A numpy array of audio amplitudes.
    :param sample_rate: Sample rate. Default is 44100 Hz.
    :param padding_duration: Duration in seconds to insert silence before and after the actual WAV data.
    :param csv: Whether to also write the WAV data to a CSV file or not.
    :return:
    """
    if padding_duration > 0:
        pad = np.zeros(sample_rate * padding_duration, dtype=np.int16)
        wav_samples = np.hstack((pad, wav_samples, pad))

    wavfile.write(file_path, sample_rate, wav_samples)

    print_c('[done] output_to_file: ' + file_path)
    pt.describe_wav(wav_samples, sample_rate)

    if csv:  # write CSV file
        with open(file_path + '.csv', 'w') as f:
            for a in wav_samples:
                f.write(str(a) + '\n')
        print_c('[done] output_to_file (csv): ' + file_path + '.csv')
