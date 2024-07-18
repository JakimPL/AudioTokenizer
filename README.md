# _AudioTokenizer

A _FastTracker2_/_Impulse Tracker_ utility for converting 16-bit mono WAV signals (44.1 kHz) into short-length-sample approximations embedded into a tracker module. 

The algorithm takes several signals (of the same shape and sample rate 44100 Hz) and combines into a single module. The explanation of the process is briefly described in `explanation.ipynb`

## Usage

To run the script, you need to have Python 3.6+ installed. The script uses the following libraries:
q
```shell
python main.py -i <input_files> -l <layers> -s <samples> -o <output_file> -t <title>
```

For example:
```shell 
python main.py -i audio/bass.wav audio/drums.wav -l 16 16 -s 32 32 -o module/drumbass.xm -t "Drum & Bass"
```

will create a _FastTracker2_ module with 16 channels per wave file, each with unique 32 samples.

## Dependencies

The script uses the following libraries:
- `numpy`
- `scipy`

Occasionally, you may want to use the `ipython` library to play a sample in a Jupyter notebook:

```python
from utils import play_sample
import scipy.io.wavfile as wav

sampling_rate, signal = wav.read("audio/bass.wav")
play_sample(signal)
```