ENABLE_COLORS = True  # set to False if having trouble with output text

# attributes
RESET = 0
BRIGHT = 1
DIM = 2
UNDERLINE = 3
BLINK = 4
REVERSE = 7
HIDDEN = 8

# color values
BLACK = 0
RED = 1
GREEN = 2
YELLOW = 3
BLUE = 4
MAGENTA = 5
CYAN = 6
WHITE = 7


def textcolor(attr=RESET, fg=WHITE, bg=BLACK):
    print('\x1B[{};{};{}m'.format(attr, fg + 30, bg + 40), end='')

# -------------------------------------------------------------------------- #


def print_b(color: int, string: str):
    """ Print bold or bright-colored text. """
    if not ENABLE_COLORS:
        print(string)
        return
    print('\x1B[1;3{}m'.format(color) + string + '\x1B[0;0m')


def print_c(color: int, string: str):
    """ Print colored text. """
    if not ENABLE_COLORS:
        print(string)
        return
    print('\x1B[3{}m'.format(color) + string + '\x1B[0m')

# -------------------------------------------------------------------------- #


if __name__ == '__main__':
    print_c(RED, 'hello!')
    print_b(RED, 'hello!')

    textcolor(BRIGHT, RED, BLACK)
    print('hello!')
    textcolor()

    print('hello!')
