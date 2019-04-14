import argparse
import logging
import time

from keras.models import load_model

import cv2
import numpy as np
import xlsxwriter

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from collections import Counter
import pickle

loaded_model = load_model('Model_4class_20personX4_TopBody.h5')

# filename = 'finalized_model60.sav'
# loaded_model = pickle.load(open(filename, 'rb'))

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
# model = load_model("conv1d_model.h5") # GANTI SESUAI NAMA MODEL
list_kelas = ["boxing", "handclapping", "handwaving", "walking"]
persons = 5  # ubah sesuai jumlah person video yang diinginkan
d_person = 4  # ubah sesuai jumlah video yang diinginkan per person

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

if __name__ == '__main__':
    data = {}
    all_true = 0
    for kelas in range(len(list_kelas)):
        count_true = 0
        for g in range(21, 21 + persons):
            for i in range(1, d_person + 1):
                hasil_kelas = []
                body_parts = []
                body_distance = []
                print('\n=== person%02d' % (g,) + '_' + list_kelas[kelas] + '_d' + str(i) + '_uncomp.avi ===\n')
                parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
                parser.add_argument('--video', type=str, default='./Videos/' + list_kelas[kelas] + '/person%02d' % (
                    g,) + '_' + list_kelas[kelas] + '_d' + str(i) + '_uncomp.avi')
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
                # logger.debug('cam read+')
                # cam = cv2.VideoCapture(args.camera)
                cap = cv2.VideoCapture(args.video)
                # ret_val, image = cap.read()
                # logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
                if (cap.isOpened() == False):
                    print("Error opening video stream or file")
                j = 0
                while (cap.isOpened()):
                    ret_val, image = cap.read()
                    if (ret_val):
                        humans = e.inference(image)
                        if len(humans) == 0:
                            body_parts.insert(j, np.zeros((2, 18)))
                        elif len(humans) > 0:
                            body_parts.insert(j, humans[0].get_coor())
                            body_distance.insert(j, get_distanceTop(body_parts[j]))
                            j += 1
                            if j == 15:  # GANTI TERGANTUNG TIMESERIES
                                data = np.asarray(body_distance)
                                #                    print('shape = ',data.shape)
                                if data.shape[0] > 15:
                                    data = data[:6]
                                #                    print('shape = ',data.shape)
                                data = data.reshape(1, 990)
                                data = np.expand_dims(data, axis=2)
                                del body_distance[0]
                                j -= 1
                                tes = loaded_model.predict(data)
                                hasil_kelas.append(np.argmax(tes))
                                print("\rPrediksi : ", np.argmax(tes),end='')
                            # image = TfPoseEstimator.draw_humans(
                            # image, humans, imgcopy=False)

                            #                logger.debug('show+')
                            # cv2.putText(image,
                            #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
                            #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            #             (0, 255, 0), 2)
                            # cv2.imshow('tf-pose-estimation result', image)
                            fps_time = time.time()
                            if cv2.waitKey(1) == 27:
                                break

                    else:
                        break
                cap.release()
                cv2.destroyAllWindows()
                # data[n] = np.asarray(body_parts)
                # n += 1
                print()
                if j > 0:
                    print("\rPrediksi Kelas : ",Counter(hasil_kelas).most_common(1)[0][0])
                    if Counter(hasil_kelas).most_common(1)[0][0] == kelas:
                        count_true += 1
            logger.debug('========== finished ==========')
        print("Kelas ", kelas, ", Jumlah benar : ", count_true)
        all_true += count_true

print("============ Automatic Testing ============")
print("Total video test : ", len(list_kelas) * persons * d_person)
print("Total benar : ", all_true)
print("Akurasi : ", str(100*all_true/(len(list_kelas) * persons * d_person)))
