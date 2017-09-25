import cv2
import numpy as np
import sys
from PIL import Image

import tensorflow as tf

from five_video_classification_methods.models import ResearchModels

import five_video_classification_methods.data

from five_video_classification_methods.extractor import Extractor

from keras import backend as K

from action_classification.ucf_labels import *

import matplotlib.pyplot as plt


class Recognizer(object):
    """docstring for Recognizer."""
    def __init__(self, checkpoint, no_of_frames):
        super(Recognizer, self).__init__()

        self.ext = Extractor()

        self.rm = ResearchModels(101, 'lstm', no_of_frames, checkpoint)

        self.f = np.random.rand(1,1024)

        self.no_of_frames = no_of_frames

        self.graph = tf.get_default_graph()

        self.counter = 0

    def run(self, frame):

        with self.graph.as_default():
            self.extract_and_stack(frame)

            return self.classify()

    def classify(self):

        # skip first frames
        if self.f.shape[0] > self.no_of_frames:

            c = self.f[0:self.no_of_frames,:][np.newaxis,:]

            # Probabilities of each action
            d = self.rm.model.predict(c)

            print "++++++++++++"
            print np.argmax(d)
            print np.amax(d)
            print str(labels[np.argmax(d)])

            #plt.scatter(self.counter, np.amax(d))

                #cv2.putText(frame, str(labels[np.argmax(d)]) + str(np.amax(d)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),1)

            # Display the resulting frame
            #cv2.imshow('frame',frame)

            #cv2.waitKey(1)

            self.counter = self.counter + 1

            return (d, labels)

    def extract_and_stack(self, frame):

        features = self.ext.extract(frame)

        self.f = np.vstack((features, self.f))
