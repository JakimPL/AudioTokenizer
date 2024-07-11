A _FastTracker2_ utility for converting 16-bit mono WAV signals (44.1 kHz) into short-length-sample approximations. 

Uses SVD to reduce the dimensionality of the signal.

The algorithm takes several signals (of the same shape and sample rate 44100 Hz) and combines into a single _FastTracker2_ module.

Usage:
```shell
python main.py -i <input_files> -l <layers> -s <samples> -o <output_file> -t <title>
```

For example:
```shell 
python main.py -i audio/bass.wav audio/drums.wav -l 16 16 -s 32 32 -o module/drumbass.xm -t "Drum & Bass"
```

will create a _FastTracker2_ module with 16 channels per wave file, each with unique 32 samples.
