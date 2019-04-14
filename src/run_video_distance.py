import argparse
import logging

import cv2
import numpy as np
import pickle

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

list_kelas = ["boxing", "handclapping", "handwaving", "walking"]
persons = 20  # ubah sesuai jumlah person video yang diinginkan
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
    for kelas in range(len(list_kelas)):
        data[kelas] = []
        for g in range(1, persons+1):
            for i in range(1, d_person+1):
                body_parts = []
                body_distance = []
                print('\n=== person%02d' % (g,)+'_' + list_kelas[kelas] + '_d' +
                      str(i)+'_uncomp.avi ===\n')
                parser = argparse.ArgumentParser(
                    description='tf-pose-estimation Video')
                parser.add_argument('--video', type=str, default='./Videos/'+list_kelas[kelas]+'/person%02d' % (
                    g,)+'_'+list_kelas[kelas]+'_d'+str(i)+'_uncomp.avi')
                parser.add_argument('--zoom', type=float, default=1.0)
                parser.add_argument('--resolution', type=str, default='432x368',
                                    help='network input resolution. default=432x368')
                parser.add_argument(
                    '--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
                parser.add_argument('--show-process', type=bool, default=False,
                                    help='for debug purpose, if enabled, speed for inference is dropped.')
                args = parser.parse_args()

        #         cap = cv2.VideoCapture('./videos/person0'+str(i)+'_handclapping_d'+str(j)+'_uncomp.avi')
        #         length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #         print(length)

                logger.debug('initialization %s : %s' %
                             (args.model, get_graph_path(args.model)))
                w, h = model_wh(args.resolution)
                e = TfPoseEstimator(get_graph_path(
                    args.model), target_size=(w, h))
                #logger.debug('cam read+')
                #cam = cv2.VideoCapture(args.camera)
                cap = cv2.VideoCapture(args.video)
                #ret_val, image = cap.read()
                #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
                if (cap.isOpened() == False):
                    print("Error opening video stream or file. Check videos path!")
                j = 0
                count_gap = 0
                while(cap.isOpened()):
                    ret_val, image = cap.read()
                    if(ret_val):
                        if count_gap<14: # Time series = 15 (frame)
                            count_gap+=1
                            continue
                        humans = e.inference(image)
                        if len(humans) > 0:
                            body_parts.insert(j, humans[0].get_coor())
                            body_distance.insert(j, get_distanceTop(body_parts[j]))
                            j += 1
        #                     image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        #                     #logger.debug('show+')
        #                     cv2.putText(image,
        #                                 "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #                                 (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                                 (0, 255, 0), 2)
        #                     cv2.imshow('tf-pose-estimation result', image)
        #                     fps_time = time.time()
        #                     if cv2.waitKey(1) == 27:
        #                         break
                    else:
                        break
                    count_gap = 0
                cap.release()
                cv2.destroyAllWindows()
                data[kelas].append(np.asarray(body_distance))
            logger.debug('finished+')

num_split = 15

d = []
l = 0
for i,e in data.items():
    for arr_vid in e:
        for j in range(len(arr_vid) - num_split):
            d1 = []
            for k in range(j, j + num_split):
                d1.append(arr_vid[k])
            d.append(np.reshape(d1,990))
            d[l] = np.append(d[l],int(i))
            l+=1
d = np.array(d,dtype='float16')

# print(d.shape)

f = open('Dataset_4class_20person_TopBody.pckl', 'wb')
pickle.dump(d, f)
f.close()