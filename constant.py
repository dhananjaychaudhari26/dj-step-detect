FS = 100.0       # sample rate, Hz
CUTOFF = 4 # try form 6 to 10      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
NYQ = 0.5 * FS  # Nyquist Frequency
ORDER = 4     # sin wave can be approx represented as quadratic
NO_OF_COLS = 40
ARTIFACT = "artifacts/all_keras.h5"