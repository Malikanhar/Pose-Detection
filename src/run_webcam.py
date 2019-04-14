import argparse
import logging
import time

from keras.models import load_model
from keras.layers import Activation

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# -- added on 13 April 2018 by Malik -- #
kelas_list = ["boxing", "handclapping", "handwaving", "walking"]
model = load_model("Model_4class_20personX4_TopBody.h5") # GANTI SESUAI NAMA MODEL
model.layers[-1].activation = Activation('sigmoid')
model.summary()
# --------------------------------------#

def get_distance(ini_list):
    list_hasil = []
    for i in range(len(ini_list[0])):
        for j in range(i+1, len(ini_list[0])):
            distance = np.sqrt(np.power((ini_list[0][i]-ini_list[0][j]),2) + np.power((ini_list[1][i]-ini_list[1][j]),2))
            list_hasil.append(distance)
    return list_hasil

''' 
0 : Hidung
1 : Dada
2 : Bahu kanan
3 : Siku kanan
4 : pergelangan tangan kanan
5 : Bahu kiri
6 : Siku kiri
7 : pergelangan tangan kiri
14 : Mata kanan
15 : Mata kiri
16 : Telinga kanan
17 : Telinga kiri
''' 
def get_distanceTop(ini_list):
    list_topbody = [0, 1 , 2, 3, 4, 5, 6, 7, 14, 15, 16, 17]
    list_hasil = []
    for i in range(len(list_topbody)):
        for j in range(i+1, len(list_topbody)):
            distance = np.sqrt(np.power((ini_list[0][list_topbody[i]]-ini_list[0][list_topbody[j]]),2) + np.power((ini_list[1][list_topbody[i]]-ini_list[1][list_topbody[j]]),2))
            list_hasil.append(distance)
    return list_hasil

fps_time = 0
if __name__ == '__main__':
    
    # -- added on 13 April 2018 by Malik -- #
    data = {}
    body_parts = []
    body_distance = []
    # --------------------------------------#

    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    if (cam.isOpened() == False):
        print("Error opening video stream or file")
    j = 0
    while True:
        ret_val, image = cam.read()
        if (ret_val):
            humans = e.inference(image)
            class_predict = []
            if len(humans) == 0:
                body_parts.insert(j, np.zeros((2, 18)))
            elif len(humans) > 0:
                body_parts.insert(j, humans[0].get_coor())

                # -- added on 13 April 2018 by Malik -- #
                body_distance.insert(j, get_distanceTop(body_parts[j]))
                j += 1
                if j == 15:  # GANTI TERGANTUNG TIMESERIES
                    data = np.asarray(body_distance).reshape(1, 990)
                    data = np.expand_dims(data, axis=2)
                    del body_distance[0]
                    j -= 1
                    class_predict = [kelas_list[np.argmax(model.predict(data)[0])]]
                    # print("Kelas :", kelas_list[np.argmax(class_predict)])
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False, kelas=class_predict)

             # --------------------------------------#

            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
        else:
            break

    cv2.destroyAllWindows()
