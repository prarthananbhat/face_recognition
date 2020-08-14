import dlib
import numpy as np
import faceblendCommon as fbc

def face_align_func(image):
    #Landmark Model Location
    # MODEL_PATH = "/Users/pbhat/The_Scool_of_AI-Phase_2/tsai-phase2/Session-03/"
    # PREDICTOR_PATH = MODEL_PATH + "shape_predictor_5_face_landmarks.dat"
    PREDICTOR_PATH = "shape_predictor_5_face_landmarks.dat"

    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

    points = fbc.getLandmarks(faceDetector,landmarkDetector,image)
    points = np.array(points,dtype=np.int32)
    img = np.float32(image)/255.0
    h = 600
    w = 600
    print(type(image))
    print(type(points))
    imnorm, points = fbc.normalizeImagesAndLandmarks((h,w),img,points)
    aligned_image = np.uint8(imnorm*255)
    return aligned_image

