# Dixi Phonetic Vectoriser

A small ML library for audio, and a python file that uses it to create a phonetic vector space.

## Dependencies

Install speech synth/ffmpeg for resampling by running `sudo apt install ffmpeg espeak mbrola mbrola-en1 mbrola-us1 mbrola-us2 mbrola-us3`. If you don't run apt, you have to find these packages. *the mbrola voices have restrictions for commercial usage, so if you're an open source purist you can remove the mb voices from synth/voices.json.*

tensorflow gpu must also be installed if you want training to finish before the end of days, but for the sake of brevity I will leave that as an exercise to the reader.

## Running the model

If you're in a hurry, I included an output log directory, so you can skip straight to Seeing the vectors.

I hope you're using **python 2**. Clone the repo and generate the training data (it has to make 280,000 small audio clips, so this may take some time);
```
git clone https://github.com/Threadnaught/dixi
cd dixi
python synthVoices.py
```

Run the model (this step takes me about 15 hours, this model is very IO-bound);
```
python phoneticVector.py
```

## Seeing the vectors
See the tensorboard output (open a web browser and go to localhost:6006);
```
tensorboard --logdir=logs
```
 ![](imgs/0.png)
 ![](imgs/1.png)
 ![](imgs/2.png)
 ![](imgs/3.png)