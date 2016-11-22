import numpy as np

print(np.random.uniform(0, 2, 20).astype(np.int16))

from _old import audio_all_clean as a

# a.output_to_speaker(a.generate_signal(5))


wsr = a.input_from_microphone(5)
a._plot_wav_analysis(*wsr, freqs=[1000, 2000])