import argparse
import logging
import time

from keras.models import load_model

import cv2
import numpy as np
import xlsxwriter

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
model = load_model("conv1d_model.h5") # GANTI SESUAI NAMA MODEL

if __name__ == '__main__':
    data = {}
    body_parts = []
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument(
        '--video', type=str, default="./videos/handwaving/person17_handwaving_d1_uncomp.avi")
# -- edited on 19 April 2018 --
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368',
                        help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str,
                        default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--s    how-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' %
                 (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    #logger.debug('cam read+')
    #cam = cv2.VideoCapture(args.camera)
    cap = cv2.VideoCapture(args.video)
    #ret_val, image = cap.read()
    #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    j = 0
    while(cap.isOpened()):
        ret_val, image = cap.read()
        if(ret_val):
            humans = e.inference(image)
            if len(humans) == 0:
                body_parts.insert(j, np.zeros((2, 18)))
            elif len(humans) > 0:
                body_parts.insert(j, humans[0].get_coor())
                j += 1
                if j == 60:  # GANTI TERGANTUNG TIMESERIES
                    data = np.asarray(body_parts).reshape(1, 2160)
                    data = np.expand_dims(data, axis=2)
                    del body_parts[0]
                    j -= 1
                    kelas = model.predict_classes(data)
                    print("Kelas: ", kelas)
                image = TfPoseEstimator.draw_humans(
                    image, humans, imgcopy=False)

                logger.debug('show+')
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
    cap.release()
    cv2.destroyAllWindows()
    # data[n] = np.asarray(body_parts)
    # n += 1
    logger.debug('finished+')
