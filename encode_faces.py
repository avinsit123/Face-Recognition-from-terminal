# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle

# import the necessary packages
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
import face_recognition
import argparse
import pickle
import cv2
import os
from os import listdir
from PIL import Image as PImage
import glob
import pickle
import sys

labels=[]
bachan_encoding=[]
avg=0

def load_images_from_folder(folder):
    images = []
    tot_images=0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        #print(filename)
        if img is not None:
            images.append(img)
            name=filename.split(".")
            labels.append(name[0])
            tot_images=tot_images+1
    return images,labels,tot_images

imgs,labels,totals=load_images_from_folder(sys.argv[1])

counter=0
for image in imgs:
    
    imageblack=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY )
    facecascade=cv2.CascadeClassifier("/Users/r17935avinash/anaconda3/pkgs/opencv3-3.1.0-py35_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
    faces=facecascade.detectMultiScale(imageblack,1.3,4)
    [(x1,y1,w1,h1)]=faces
    bachan_encoding.append(np.array(face_recognition.face_encodings(image[y1:y1+h1,x1:x1+w1])))
    cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(255,0,0),4)
    print("Image",counter+1,"encoded. Label:",labels[counter])
    counter=counter+1;
    #plt.imshow(image)
    #plt.xlabel(labels)
    plt.show()
    
with open(sys.argv[1]+'/image_encodings','wb') as file1:
    pickle.dump(bachan_encoding,file1)
with open(sys.argv[1]+'/labels','wb') as file2:
    pickle.dump(labels,file2)
file1.close()
file2.close()

print("Successfully trained")
#for i in range(totals):
       #print(np.sum((bachan_encoding[i]-avg)**2,axis=-1))

#'/Users/r17935avinash/Desktop/facerecog/Amitabh/image_encodings'
#facecascade=cv2.CascadeClassifier("/Users/r17935avinash/anaconda3/pkgs/opencv3-3.1.0-py35_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
#faces=facecascade.detectMultiScale(image,1.5,2)
#[(x1,y1,w1,h1)]=faces_1
#v2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(255,0,0),4)
#cv2.rectangle(image_2,(x1,y1),(x1+w1,y1+h1),(255,0,0),4)
#bachan_encoding_1=(np.array)(face_recognition.face_encodings(image_1[y1:y1+h1,x1:x1+w1]))
#bachan_encoding_2=(np.array)(face_recognition.face_encodings(image_2[y2:y2+h2,x2:x2+w2]))




