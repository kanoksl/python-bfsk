import random
import os.path

import audio_io as io
import audio_modem as md
import audio_processing as ap
import configuration as cf
import digital_data as digital
import plotter as pt
from digital_data import BIT0, BIT1

# -------------------------------------------------------------------------- #

TESTCASE_FOLDER = 'testcases/'  # default folder of the testcase files

# -------------------------------------------------------------------------- #


def testcase_name(config, data_len):
    """
    Get the testcase file name (.txt) for the given configuration.

    :param config: Modem configuration.
    :param data_len: Length of data in bits.
    :return: The testcase file name.
    """
    filename = '{}_{}bps_[{};{}]_{}bit.txt'.format(config.mode, config.bit_rate, config.f0, config.f1, data_len)
    return filename


def read_testcase(file_path):
    """
    Read the modem configuration and data from a testcase file.

    :param file_path: Path of the testcase text file.
    :return: A tuple (config, data).
    """
    print('reading testcase \'{}\'...'.format(file_path))
    with open(file_path, 'r') as f:
        mode = f.readline().strip()
        if mode == cf.BFSK:
            br = int(f.readline().strip())
            f0 = int(f.readline().strip())
            f1 = int(f.readline().strip())
            data = f.readline().strip()
            return cf.Config(mode, br, f0, f1), data
        else:
            raise NotImplementedError('unsupported modulation/encoding scheme')


def generate_testcase_audio(file_path, padding=1):
    """
    Generate WAV file for a testcase.

    :param file_path: Path of the testcase file (.txt).
    :param padding: Length in seconds.
    """
    config, data = read_testcase(file_path)
    modem = md.ModulatorDemodulator(config)

    bits = digital.encode(data)
    wav, sr = modem.modulate(bits)

    file_path += '.wav'
    io.output_to_file(file_path, wav, sr, padding_duration=padding)
    print('finished generating audio for a testcase: ' + file_path)


def generate_testcase(data_length, config, file_path='', audio=True):
    """
    Generate a testcase text file with the given configuration.

    :param data_length: Length of the random data in bits.
    :param config: Modem configuration.
    :param file_path: Path of the testcase text file.
    :param audio: Whether to create the corresponding WAV file or not.
    """
    if not file_path:
        file_path = TESTCASE_FOLDER + testcase_name(config, data_length)

    # generate random data
    data = list()
    for i in range(data_length):
        data.append(BIT0 if random.random() < 0.5 else BIT1)
    data = ''.join(data)

    # write the text file
    with open(file_path, 'w') as f:
        f.write('{}\n'.format(config.mode))
        f.write('{}\n'.format(config.bit_rate))
        f.write('{}\n'.format(config.f0))
        f.write('{}\n'.format(config.f1))
        f.write(data)

    print('finished generating a testcase: ' + file_path)

    if audio:  # create WAV file
        generate_testcase_audio(file_path)


def run_testcase(file_path, duration=0, use_file=False):
    """
    Run a testcase. Test whether the transmitted data is identical to the original data.

    :param file_path: Path of the testcase text file.
    :param duration: Duration to record the audio from the microphone.
    :param use_file: Use the testcase's WAV file instead of new recording.
    """
    print('running testcase \'{}\'...\n\n'.format(file_path))
    config, data = read_testcase(file_path)
    modem = md.ModulatorDemodulator(config)
    print('\n')

    if duration <= 0:  # auto determine recording duration
        duration = 5 + (len(data) / config.bit_rate)

    if not use_file:
        wav, sr = io.input_from_microphone(duration)
        io.output_to_file(TESTCASE_FOLDER + 'last_recorded.wav', wav, sr)
    else:
        wav, sr = io.input_from_file(file_path + '.wav')
    print('\n')

    bits = modem.demodulate(wav, sr)
    bits = digital.decode(bits)
    bits = ''.join(bits)

    print('data: ' + data)
    print('recv: ' + bits)
    print('diff: ', end='')

    error = 0  # compare and check for errors
    for i in range(min(len(bits), len(data))):
        if str(bits[i]) != str(data[i]):
            print(bits[i], end='')
            error += 1
        else:
            print('_', end='')
    print()

    if len(bits) != len(data):
        print('ERROR: length mismatch')
    print('original data length = {:,d} bits'.format(len(data)))
    print('received (decoded)   = {:,d} bits'.format(len(bits)))
    print('bit error: {:,d} bits ({:.2f} %)'.format(error, error / len(data) * 100))
    print('\n')

# -------------------------------------------------------------------------- #


def begin():

    # configurations - set the following values before executing. notes:
    #  - bit_rate should be a divisor of sample rate (44,100 hz) to avoid
    #    errors due to rounding.
    #  - frequencies f0 and f1 must be at least twice the bit_rate.

    data_length = 400
    bit_rate = 2205
    f0 = 11025
    f1 = 8820
    easy_mode = False

    # ---------------------------------------------------------------------- #

    config = cf.Config(cf.BFSK, bit_rate, f0, f1)
    testcase_file = TESTCASE_FOLDER + testcase_name(config, data_length)

    # generate a new testcase file if not already exist
    if not os.path.isfile(testcase_file) or not os.path.isfile(testcase_file + '.wav'):
        generate_testcase(data_length, config, audio=True)

    # analyze the generated audio
    wav, sr = io.input_from_file(testcase_file + '.wav')
    # print('plotting the generated audio file')
    # pt.plot_wav_analysis(wav, sr, [f0, f1], dft=True)

    run_testcase(testcase_file, duration=0, use_file=easy_mode)


if __name__ == '__main__':
    begin()
