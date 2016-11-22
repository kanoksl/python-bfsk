
import util
def print_b(string): util.print_b(util.RED, string)
def print_c(string): util.print_c(util.RED, string)

# -------------------------------------------------------------------------- #

# constants for bit values
BIT0 = '0'
BIT1 = '1'

# maximum number of consecutive 1s allowed in data bits
STUFFING_LEN = 31

# signals the beginning/ending of the data stream
FLAG_SEQUENCE = ([BIT1] * (STUFFING_LEN + 1))
FLAG_LEN = len(FLAG_SEQUENCE)


def encode(d_stream):
    """
    Encode the data bits, adding start/end sequence and perform bit stuffing.

    :param d_stream: List of '0's and '1's.
    :return: Encoded list of '0's and '1's. Ready to be transmitted.
    """
    t_stream = list(d_stream)

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

    print(t_stream)
    return t_stream


def decode(t_stream):
    """
    Decode the transmitted bits, removing start/end sequence and bit stuffing.

    :param t_stream: Encoded list of '0's and '1's.
    :return: List of '0's and '1's, the actual data.
    """
    d_stream = list(t_stream)

    if not (d_stream[:FLAG_LEN] == FLAG_SEQUENCE):
        print_b('[WARNING]: missing start flag sequences')
        print_c('  first {} bits: '.format(FLAG_LEN) + ''.join(d_stream[:FLAG_LEN]) + '...')
    if not (FLAG_SEQUENCE == d_stream[-FLAG_LEN:]):
        print_b('[WARNING]: missing end flag sequences')
        print_c('  last {} bits: ...'.format(FLAG_LEN) + ''.join(d_stream[-FLAG_LEN:]))

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
                print_c('[ERROR]: incorrect bit stuffing pattern at pos: ' + str(j))
        i = j + 1

    return d_stream
