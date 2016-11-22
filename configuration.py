# -------------------------------------------------------------------------- #

# Use 16-bit PCM format for WAV samples; values range from -32768 to +32767.
# Use numpy's ndarray with dtype='int16'. The axis should be 1 (horizontal).

# Most WAV-related functions will return both WAV array and sample rate,
# even if the sample rate is given as an argument (for consistency).

# -------------------------------------------------------------------------- #

DEFAULT_SAMPLE_RATE = 44100  # integer, Hz
DEFAULT_AMPLITUDE = 10000  # in range [0, 32767]

# -------------------------------------------------------------------------- #

BFSK = 'bfsk'
MFSK = 'mfsk'


class Config:
    """ A simple modem configuration. Currently support only BFSK. """

    def __init__(self, mode, bit_rate, f0, f1):
        assert mode == BFSK
        self.mode = mode
        self.bit_rate = bit_rate
        self.f0 = f0
        self.f1 = f1

    def __str__(self):
        description = 'Config { mode={}, bitrate={:,d}bps, f0={:,d}hz, f1={:,d}hz }'
        return description.format(self.mode, self.bit_rate, self.f0, self.f1)
