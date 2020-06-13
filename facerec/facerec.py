#!/usr/bin/python
# 
#   The FaceRec class provides several tools for the complete face recognition process.
#       1. Face detection
#       2. Landmark detection
#       3. Embedding computation
#       4. Embeddings comparison
#
#   USAGE:   
#       - See the simplest_example.py code for a getting started.
#       - For more complete examples see fr.py.
# 

import numpy as np
import cv2 as cv
import dlib


class FaceRec:
    def __init__(self, resources_path):
        """ Class constructor

        Params:
            resources_path: relative or full path to the resources folder, e.g., ./resources

        Returns:
            FaceRec: an instance of the FaceRec class
        """

        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_p = dlib.shape_predictor(resources_path +
                                            '/shape_predictor_5_face_landmarks.dat')
        self.face_model = dlib.face_recognition_model_v1(resources_path +
                                                         '/dlib_face_recognition_resnet_model_v1.dat')
        self.face_cascade = cv.CascadeClassifier(resources_path +
                                                 '/haarcascade_frontalface_default.xml')

    def face_embedding(self, image, return_aligned=False):
        """ Function that runs a complete process for obtaning a face embedding
        from an input image.
        The process includes detection->landmarks->alignment->descriptor.

        Params:
            image:      input image, e.g., image loaded from file, read from camera, etc.

        Returns:
            np.array:   128-D np.array with embedding.
            empty-list: if no face found in the image.
        """

        # search for faces in the image
        faces = self.detect_faces(image)

        if len(faces) > 0:
            # if more than one face found,
            # apply some heuristic to select just one
            # here we are using the biggest found face
            face = faces[np.argmax([w*h for (_, _, w, h) in faces])]

            # detect landmarks
            landmarks = self.detect_landmarks(image, face)

            # align face
            aligned = self.aligned_face(image, landmarks)

            # compute descriptor, this includes alignment using landmarks
            descr = self.compute_descriptor_for_image(aligned)

            if return_aligned:
                return np.array(descr).reshape(1, -1), aligned
            else:
                return np.array(descr).reshape(1, -1)

        # print('ERROR: No faces found in input image')
        if return_aligned:
            return [], []
        else:
            return []


    def face_distance(self, descr_1, descr_2):
        """ Function for computing distances among embeddings

        Params:
            descr_1:   an embedding or array of embeddings
            descr_2:   an embedding or array of embeddings
            SEE examples for details.

        Returns:
            Euclidean distances between embeddings.
        """

        if len(descr_1) == 0 or len(descr_2) == 0:
            return np.empty((0))

        return np.linalg.norm(descr_1 - descr_2, axis=1)


    def detect_faces(self, image):
        """ Function for detecting faces in an image

        Params:
            image: input image, e.g., image loaded from file, read from camera, etc.

        Returns:
            np.array:   an array containing all the detected faces (rect), each rect per row
                        in the format x, y, w, h.
            empty-list: if no face found in the image.
        """

        # if is a BGR image, convert to gray
        if image.shape[2] == 3:
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        # if is already gray, leave as is
        elif image.shape[2] == 3:
            gray = image
        else:
            print('ERROR in image color space. Image shape is', image.shape)
            return None

        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return []

        return faces


    def detect_landmarks(self, image, roi):
        """ Function for detecting landmarks in a face image

        Params:
            image: input image, e.g., image loaded from file, read from camera, etc.
            roi:   rectangle specifiying face location in image

        Returns:
            landmarks array
        """

        roi_dlib = dlib.rectangle(roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3])
        landmarks = self.shape_p(image, roi_dlib)
        return landmarks


    def compute_descriptor_for_image(self, image, landmarks=None):
        """ Function for computing descriptor (embedding) for a face image

        Params:
            image:      input image, e.g., image loaded from file, read from camera, etc.
            landmarks:  landmarks extracted from the face image, which are used for alignment.

        Returns:
            np.array:   128-D np.array with embedding.
        """
        
        if landmarks is not None:
            descr = self.face_model.compute_face_descriptor(image, landmarks)
        else:
            descr = self.face_model.compute_face_descriptor(image)

        return descr


    def aligned_face(self, image, landmarks):
        return dlib.get_face_chip(image, landmarks)
        

