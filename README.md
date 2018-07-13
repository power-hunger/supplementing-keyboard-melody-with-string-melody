# Classical Melody Generator

This project allows you to train a neural network on two instrument midi files. Afterwards new melody can be generated 
to one instrument solo melody. 

## Requirements

* Python 3.x
* Installing the following packages using pip:
	* Music21
	* Keras
	* Tensorflow
	* h5py

## Training

To train the network you run **lstm.py**.

E.g.

```
python lstm.py
```

The network will use every midi file in ./midi_songs to train the network. The midi files must contain two known 
instruments with similar note amount. Default instruments are- any string instrument and any keyboard instrument.

**NOTE**: You can stop the process at any point in time and the weights from the latest completed epoch will be available for text generation purposes.

## Generating music

Once you have trained the network you can generate text using **predict.py**

E.g.

```
python predict.py
```

You can run the prediction file right away using the **weights.hdf5** file

## Prepare your own song data set

To prepare your own data set put your midi files in /midi_songs and then run **prepareDataset.py**.

E.g.

```
python prepareDataset.py
```

## Result

Result can be seen folowing this link - https://www.youtube.com/playlist?list=PLHrS8dlHp_NDW01b2cS8a43qtOiLVu7RK
