# Face Recognition from terminal
Gone are the days of training long Convolutional Resnets,this repository allows you to perform face recognition and verification from the terminal/command line itself.

The face recognizer is built using [dlib](http://dlib.net/)'s state-of-the-art face recognition
built with deep learning. The model has an accuracy of 99.38% on the
[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark.The neural 
network used is the [inception resnet V2](https://medium.com/@williamkoehrsen/facial-recognition-using-googles-convolutional-neural-network-5aa752b4240e)(or popularly known as GoogLeNet).

## Prerequisites

The following python libraries need to be preinstalled to for the face-recognizer to work 

 Dlib
```
pip install dlib
```

 [face_recognition](https://github.com/ageitgey/face_recognition)

```
pip install face_recognition
```

 OpenCV

```
pip install opencv_python //for main modules
```
 
The code will work on any python ide,however python distributions like Anaconda would be preferred as they have all the necessary modules pre-installed already.



