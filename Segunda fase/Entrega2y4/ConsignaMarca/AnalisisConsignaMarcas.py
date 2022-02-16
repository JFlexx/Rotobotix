
import argparse
import numpy as np
import re
import sys
import time # noqa, disable flycheck warning

from matplotlib import pyplot as plt # noqa, disable flycheck warning
from os import listdir, mkdir
from os.path import isfile, join
from imageio import imread, imsave
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier

import cv2
import select_pixels as sel


VIDEO = 'video2017-3.avi'
IMAGENES = 'imgs'
SEGM_DIR = 'SegmFrames'

ANALY_DIR = 'ImagenesConsigna'
CHULL_DIR = 'ChullFrames'
VID_DIR = 'OutputVideos'



#Constantes
MARKS = ['Cruz', 'Escalera', 'Persona', 'Telefono']

FONT = cv2.FONT_HERSHEY_SIMPLEX#constante que servira para utilizar la fuente de texto en el video que se muestre


neigh_clf = None


class TypeObjAutomaton:

    def __init__(self):
        self.state = 0

    def _state(self):
        return self.state

    def _reset(self):
        if self.state:
            self.state = 0

    def __decrease(self):
        self.state -= 1
        return 0 if (self.state < 0) else 1

    def __increase(self):
        self.state += 1
        return 0 if (self.state < 0) else 1

    def getType(self, state):
        if state:
            return self.__decrease()
        else:
            return self.__increase()


class MarkAutomaton:
    # States are 0, 1, 2, 3
    # Get the maximum state
    def __init__(self):
        self.state = [0, 0, 0, 0]

    def _state(self):
        return self.state

    def _reset(self):
        if not all(self.state):
            self.state = [0, 0, 0, 0]

    def getType(self, pred):
        self.state[pred] += 1
        return MARKS[np.argmax(self.state)]


def marking():
    capture = cv2.VideoCapture(VIDEO)
    count = 0

    make_dir(TRAIN_DIR)

    pause = False
    while(capture.isOpened()):
        if not pause:
            ret, frame = capture.read()
        if ret and not count % 24:
            cv2.imshow('Image', frame)

            # compare key pressed with the ascii code of the character
            key = cv2.waitKey(1000)

            # (n)ext image
            if (key & 0xFF) == ord('n'):
                count += 1
                continue

            # mark image, (s)top
            if (key & 0xFF) == ord('s'):
                # change from BGR to RGB format
                im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # mark training pixels
                mark_img = sel.select_fg_bg(im_rgb)

                imsave(join(TRAIN_DIR, 'OriginalImg'+str(count)+'.png'), im_rgb)
                imsave(join(TRAIN_DIR, 'TrainingImg'+str(count)+'.png'), mark_img)

            # (q)uit program
            if (key & 0xFF) == ord('q'):
                break

            # (p)ause program
            if (key & 0xFF) == ord('p'):
                pause = not pause

        elif not ret:
            print ("End of video")
            break

        count += 1

    capture.release()
    cv2.destroyAllWindows()


# mark and train_img_m params in case of training knn classifier (marks)
# train with a different training image
def training(mark=False, train_img_m=''):
 

    # Height x Width x channel
    orig_img = imread(join(IMAGENES, 'imagenOrigen.png'))
    mark_img = imread(join(IMAGENES, 'imagenMarcada.png'))
	
	#normalizamos
    img_norm = np.rollaxis((np.rollaxis(orig_img, 2))/(np.sum(orig_img, 2)), 0, 3)

    data_red = img_norm[np.where(np.all(np.equal(mark_img, (255, 0, 0)), 2))]
    data_green = img_norm[np.where(np.all(np.equal(mark_img, (0, 255, 0)), 2))]
    data_blue = img_norm[np.where(np.all(np.equal(mark_img, (0, 0, 255)), 2))]
    
    	#normalizamos otra vez
    img_norm = np.rollaxis((np.rollaxis(orig_img, 2))/(np.sum(orig_img, 2)), 0, 3)

    data_red = img_norm[np.where(np.all(np.equal(mark_img, (255, 0, 0)), 2))]
    data_green = img_norm[np.where(np.all(np.equal(mark_img, (0, 255, 0)), 2))]
    data_blue = img_norm[np.where(np.all(np.equal(mark_img, (0, 0, 255)), 2))]

    data = np.concatenate([data_red, data_green, data_blue])#obtemos

    target = np.concatenate([np.zeros(len(data_red[:]), dtype=int),
                             np.ones(len(data_green[:]), dtype=int),
                             np.full(len(data_blue[:]), 2, dtype=int)])

    clf = NearestCentroid()
    clf.fit(data, target)
    return clf


# mark param to segmentate all the image, not just the 90: pixels
# segm param to show a frame with the segmentated image
def segmentation(clf, frame, count, args, segm, mark=False):
    if not mark:
        shape = frame[90:, :].shape
        frame_rgb = cv2.cvtColor(frame[90:, :], cv2.COLOR_BGR2RGB)
    else:
        shape = frame.shape  # Segm all
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Segm all

    shape = frame.shape  # Segm all
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Segm all

    img_norm = np.rollaxis((np.rollaxis(frame_rgb, 2)+0.1)/(np.sum(frame_rgb, 2)+0.1), 0, 3)



    # Reshape in order to reduce the 3-dimensional array to 1-dimensional (needed for predict)
    reshape = img_norm.reshape(shape[0]*shape[1], 3)
    labels = clf.predict(reshape)

    # Reshape back, from 1-dimensional to 2-dimensional
    reshape_back = labels.reshape(shape[0], shape[1])

    paleta = np.array([[255, 0, 0], [0, 0, 0], [0, 0, 255]], dtype=np.uint8)

    # Automatic reshape is being done here, from 2-dimensional to 3-dimensional array [[1, 1, ...]] -> [[[0,0,0], ....]]
    aux = paleta[reshape_back]

    segm_img = cv2.cvtColor(aux, cv2.COLOR_RGB2BGR)

    if segm:
        cv2.imshow('Segm', segm_img)

    if args.genVideo:
        if args.genVideo == 'segm':
            cv2.imwrite(join(SEGM_DIR, 'SegmImg'+str(count)+'.png'), segm_img)

    # Image with the line in white
    line_img = (reshape_back == 2).astype(np.uint8)*255

    # Image with the arrow/mark in white
    arrow_mark_img = (reshape_back == 0).astype(np.uint8)*255

    return line_img, arrow_mark_img


def analysis(clf, args, segm=False):
    if VIDEO == '0':
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(VIDEO)
    count = 0
    latest_org = 0

    if args.genVideo:
        if args.genVideo == 'segm':
            make_dir(SEGM_DIR)
        elif args.genVideo == 'analy':
            make_dir(ANALY_DIR)
        elif args.genVideo == 'chull':
            make_dir(CHULL_DIR)

    pause = False
    type_aut = TypeObjAutomaton()
    mark_aut = MarkAutomaton()
    while(capture.isOpened()):
        if not pause:
            ret, frame = capture.read()
        # if ret and not count % 24:
        if ret:
            # if video == '0':
            #     ret = capture.set(3, 340)
            #     ret = capture.set(240)
            cv2.imshow('Original', frame)

            line_img, arrow_mark_img = segmentation(clf, frame, count, args, segm)

            # FindContours is destructive, so we copy make a copy
            line_img_cp = line_img.copy()

            # FindContours is destructive, so we copy make a copy
            arrow_mark_img_cp = arrow_mark_img.copy()

            # Should we use cv2.CHAIN_APPROX_NONE? or cv2.CHAIN_APPROX_SIMPLE? the former stores all points, the latter stores the basic ones
            # Find contours of line
            cnts_l, hier = cv2.findContours(line_img, cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_NONE)

            # Find contours of arror/mark
            cnts_am, hier = cv2.findContours(arrow_mark_img, cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_NONE)

            # Removes small contours, i.e: small squares
            newcnts_l = [cnt for cnt in cnts_l if len(cnt) > 100]
            newcnts_am = [cnt for cnt in cnts_am if len(cnt) > 75]

            # DrawContours is destructive
            # analy = frame.copy()[90:]
            analy = frame.copy()

            # Return list of indices of points in contour
            chull_list_l = [cv2.convexHull(cont, returnPoints=False) for cont in newcnts_l]
            chull_list_am = [cv2.convexHull(cont, returnPoints=False) for cont in newcnts_am]

            conv_defs_l = [(cv2.convexityDefects(cont, chull), pos) for pos, (cont, chull) in
                           enumerate(zip(newcnts_l, chull_list_l)) if len(cont) > 3 and len(chull) > 3]

            conv_defs_am = [(cv2.convexityDefects(cont, chull), pos) for pos, (cont, chull) in
                            enumerate(zip(newcnts_am, chull_list_am)) if len(cont) > 3 and len(chull) > 3]

            list_conv_defs_l = []
            list_cont_l = []
            list_conv_defs_am = []
            list_cont_am = []
            # Only save the convexity defects whose hole is larger than ~4 pixels (1000/256).
            for el in conv_defs_l:
                if el is not None:
                    aux = el[0][:, :, 3] > 1000
                    if any(aux):
                        list_conv_defs_l.append(el[0][aux])  # el = (convDefs, position)
                        list_cont_l.append(newcnts_l[el[1]])

            for el in conv_defs_am:
                if el is not None:
                    aux = el[0][:, :, 3] > 1000
                    if any(aux):
                        list_conv_defs_am.append(el[0][aux])
                        list_cont_am.append(newcnts_am[el[1]])

            mark = True

            if not list_conv_defs_l:
                cv2.putText(analy, "Linea recta", (0, 140),
                            FONT, 0.5, (0, 0, 0), 1)

            for pos, el in enumerate(list_conv_defs_l):
                for i in range(el.shape[0]):
                    if el.shape[0] == 1:
                        # [NormX, NormY, PointX, PointY]
                        [vx, vy, x, y] = cv2.fitLine(list_cont_l[pos], cv2.DIST_L2, 0, 0.01, 0.01)
                        slope = vy/vx
                        if slope > 0:
                            cv2.putText(analy, "Giro izq", (0, 140),
                                        FONT, 0.5, (0, 0, 0), 1)
                        else:
                            cv2.putText(analy, "Giro dcha", (0, 140),
                                        FONT, 0.5, (0, 0, 0), 1)

                    elif el.shape[0] == 2 or el.shape[0] == 3:
                        cv2.putText(analy, "Cruce 2 salidas", (0, 140),
                                    FONT, 0.5, (0, 0, 0), 1)
                        mark = False
                    elif el.shape[0] == 4:
                        cv2.putText(analy, "Cruce 3 salidas", (0, 140),
                                    FONT, 0.5, (0, 0, 0), 1)
                        mark = False

                    if args.genVideo and args.genVideo == 'chull':
                        # Draw convex hull and hole
                        s, e, f, d = el[i]
                        start = tuple(list_cont_l[pos][s][0])
                        end = tuple(list_cont_l[pos][e][0])
                        far = tuple(list_cont_l[pos][f][0])
                        cv2.line(analy, start, end, [0, 255, 0], 2)
                        cv2.circle(analy, far, 3, [0, 0, 255], -1)

            if args.genVideo and args.genVideo == 'chull':
                for pos, el in enumerate(list_conv_defs_am):
                    for i in range(el.shape[0]):
                        # Draw convex hull and hole
                        s, e, f, d = el[i]
                        start = tuple(list_cont_am[pos][s][0])
                        end = tuple(list_cont_am[pos][e][0])
                        far = tuple(list_cont_am[pos][f][0])
                        cv2.line(analy, start, end, [0, 255, 0], 2)
                        cv2.circle(analy, far, 3, [0, 0, 255], -1)

            if not newcnts_am:
                type_aut._reset()
                mark_aut._reset()
            else:
                if not type_aut.getType(mark):
                    if len(newcnts_am) == 1:
                        hu_mom = cv2.HuMoments(cv2.moments(newcnts_am[0])).flatten()
                        # hu_mom2 = -np.sign(hu_mom)*np.log10(np.abs(hu_mom))
                        pred = neigh_clf.predict([hu_mom])
                        if pred == 0:
                            cv2.putText(analy, "Cruz", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        elif pred == 1:
                            cv2.putText(analy, "Escalera", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        elif pred == 2:
                            cv2.putText(analy, "Persona", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        elif pred == 3:
                            cv2.putText(analy, "Telefono", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                else:
                    cv2.putText(analy, "Flecha", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    for c in newcnts_am:
                        ellipse = cv2.fitEllipse(c)
                        center, axis, angle = ellipse

                        # Axis angles, major, minor
                        maj_ang = np.deg2rad(angle)
                        min_ang = maj_ang + np.pi/2

                        # Axis lenghts
                        major_axis = axis[1]
                        minor_axis = axis[0]

                        # Lines of axis, first line and his complementary
                        lineX1 = int(center[0]) + int(np.sin(maj_ang)*(major_axis/2))
                        lineY1 = int(center[1]) - int(np.cos(maj_ang)*(major_axis/2))
                        lineX2 = int(center[0]) - int(np.sin(maj_ang)*(major_axis/2))
                        lineY2 = int(center[1]) + int(np.cos(maj_ang)*(major_axis/2))

                        if args.genVideo and args.genVideo == 'chull':
                            linex1 = int(center[0]) + int(np.sin(min_ang)*(minor_axis/2))
                            liney1 = int(center[1]) - int(np.cos(min_ang)*(minor_axis/2))
                            cv2.line(analy, (int(center[0]), int(center[1])), (lineX1, lineY1), (0, 0, 255), 2)
                            cv2.line(analy, (int(center[0]), int(center[1])), (linex1, liney1), (255, 0, 0), 2)
                            cv2.circle(analy, (int(center[0]), int(center[1])), 3, (0, 0, 0), -1)
                            cv2.ellipse(analy, ellipse, (0, 255, 0), 2)
                            cv2.putText(analy, "Ang. elipse: "+str(angle), (0, 110),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                        # Get coordinates of arrow pixels
                        # arrow = cv2.findNonZero(arrow_mark_img_cp)[:, 0, :]
                        idx = np.where(arrow_mark_img_cp != 0)
                        size_idx = len(idx[0])
                        arrow = np.array([[idx[1][idy], idx[0][idy]] for idy in range(size_idx)])
                        angle360 = angle        # Initial angle in [0,180)
                        if 45 <= angle <= 135:  
                            # divido la flecha en 2, dependiendo de las coordenadas obtenidas del centro de la flecha
                            left = [1 for px in arrow if px[0] < center[0]]
                            right = [1 for px in arrow if px[0] > center[0]]
                            if len(right) >= len(left):
                                peak = (lineX1, lineY1)  # Arrow peak is the point in major axis 1
                            else:
                                peak = (lineX2, lineY2)  # Arrow peak is the point in major axis 2
                                angle360 += 180          # Real angle in [0,360)
                        else:  
                            up = [1 for px in arrow if px[1] < center[1]]
                            down = [1 for px in arrow if px[1] > center[1]]
                            if (len(up) >= len(down) and angle < 45) or (len(down) >= len(up) and angle > 135):
                                peak = (lineX1, lineY1)  # Arrow peak is the point in major axis 1
                            else:
                                peak = (lineX2, lineY2)  # Arrow peak is the point in major axis 2
                                angle360 += 180

                        angle360 = int(angle360)
                        hasLine = 1
                        if angle360 >= 337.5 or angle360 < 22.5:
                            cv2.putText(analy, "Norte (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            lineDistance = 0
                        elif angle360 < 67.5:
                            cv2.putText(analy, "Noreste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            lineDistance = -0.25
                        elif angle360 < 112.5:
                            cv2.putText(analy, "Este (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            lineDistance = -0.5
                        elif angle360 < 157.5:
                            cv2.putText(analy, "Sureste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            lineDistance = -0.8
                        elif angle360 < 202.5:
                            cv2.putText(analy, "Sur (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            lineDistance = 1
                        elif angle360 < 247.5:
                            cv2.putText(analy, "Suroeste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            lineDistance = 0.8
                        elif angle360 < 292.5:
                            cv2.putText(analy, "Oeste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            lineDistance = 0.5
                        elif angle360 < 337.5:
                            cv2.putText(analy, "Noroeste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            lineDistance = 0.25

                        cv2.line(analy, (int(peak[0]), int(peak[1])), (int(center[0]), int(center[1])), (0, 0, 255), 2)
                        cv2.circle(analy, (int(peak[0]), int(peak[1])), 3, (0, 255, 0), -1)

            left_border = line_img_cp[:, :20].copy()
            right_border = line_img_cp[:, 300:].copy()
            top_border = line_img_cp[:20, 20:300].copy()
            bot_border = line_img_cp[220:, 20:300].copy()

            all_mlc = []
            all_mrc = []
            all_mtc = []
            all_mbc = []

            left_cnt, hier = cv2.findContours(left_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            left_cnt = [cnt for cnt in left_cnt if cv2.contourArea(cnt) > 50]
            if left_cnt:
                for l in left_cnt:
                    mlc = np.mean(l[:, :, :], axis=0, dtype=np.int32)
                    all_mlc.append(mlc)
                    cv2.circle(analy, (mlc[0, 0], mlc[0, 1]), 3, (0, 255, 0), -1)

            right_cnt, hier = cv2.findContours(right_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            right_cnt = [cnt for cnt in right_cnt if cv2.contourArea(cnt) > 50]
            if right_cnt:
                for r in right_cnt:
                    r[:, :, 0] = r[:, :, 0] + 300
                    mrc = np.mean(r[:, :, :], axis=0, dtype=np.int32)
                    all_mrc.append(mrc)
                    cv2.circle(analy, (mrc[0, 0], mrc[0, 1]), 3, (0, 255, 0), -1)

            top_cnt, hier = cv2.findContours(top_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            top_cnt = [cnt for cnt in top_cnt if cv2.contourArea(cnt) > 50]
            if top_cnt:
                for t in top_cnt:
                    t[:, :, 0] = t[:, :, 0] + 20
                    mtc = np.mean(t[:, :, :], axis=0, dtype=np.int32)
                    all_mtc.append(mtc)
                    cv2.circle(analy, (mtc[0, 0], mtc[0, 1]), 3, (0, 255, 0), -1)

            bot_cnt, hier = cv2.findContours(bot_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            bot_cnt = [cnt for cnt in bot_cnt if cv2.contourArea(cnt) > 50]
            if bot_cnt:
                for b in bot_cnt:
                    b[:, :, 0] = b[:, :, 0] + 20
                    b[:, :, 1] = b[:, :, 1] + 220
                    mbc = np.mean(b[:, :, :], axis=0, dtype=np.int32)
                    all_mbc.append(mbc)
                    cv2.circle(analy, (mbc[0, 0], mbc[0, 1]), 3, (255, 0, 255), -1)

            if args.genVideo and args.genVideo == 'chull':
                cv2.drawContours(analy, left_cnt, -1, (255, 0, 0), 2)
                cv2.drawContours(analy, right_cnt, -1, (0, 255, 0), 2)
                cv2.drawContours(analy, top_cnt, -1, (255, 0, 255), 2)
                cv2.drawContours(analy, bot_cnt, -1, (0, 0, 255), 2)

            cv2.drawContours(analy, newcnts_l, -1, (255, 0, 0), 1)
            cv2.drawContours(analy, newcnts_am, -1, (0, 0, 255), 1)
            
            cv2.imshow("ConsignaYContornos", analy) 


            #Esto es para guardar las imagenes del analisis(lo comento sino peta de tantas fotos)#
            #if args.genVideo:
            #     if args.genVideo == 'analy':
            #         cv2.imwrite(join(ANALY_DIR, 'AnalyImg'+str(count)+'.png'), analy)
            #     elif args.genVideo == 'chull':
            #         cv2.imwrite(join(CHULL_DIR, 'ChullImg'+str(count)+'.png'), analy)

            # compare key pressed with the ascii code of the character
            key = cv2.waitKey(10)

            # (n)ext image
            if (key & 0xFF) == ord('n'):
                count += 1
                continue

            # (q)uit program
            if (key & 0xFF) == ord('q'):
                break

            # (p)ause program
            if (key & 0xFF) == ord('p'):
                pause = not pause

        elif not ret:
            print ("End of video")
            break

        count += 1

    capture.release()
  
    cv2.destroyAllWindows()


def mark_train(args):
    all_hu = np.load('moments.hu')
    labels = np.load('moments.labels')

    q_n = 1
    cov_list = np.cov(all_hu.reshape(400, 7).T)
    neigh = KNeighborsClassifier(n_neighbors=q_n, weights='distance',
                                 metric='mahalanobis', metric_params={'V': cov_list})

    n_images = 4*100
    neigh.fit(all_hu.reshape((n_images, 7)), labels.reshape((n_images,)))
    global neigh_clf
    neigh_clf = neigh
    return neigh


def gen_video(name, procedure):
    make_dir(VID_DIR)

    aux_dir = None

    if procedure == 'analy':
        aux_dir = ANALY_DIR
    elif procedure == 'chull':
        aux_dir = CHULL_DIR

    images = [f for f in listdir(aux_dir) if isfile(join(aux_dir, f))]

    if not len(images):
        print ("No images to create the video.")
        sys.exit()

    images = natural_sort(images)
    aux = cv2.imread(join(aux_dir, images[0]))

    height, width, layers = aux.shape

    video = cv2.VideoWriter(join(VID_DIR, name+'.avi'), cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 1.0, (width, height))

    for img in images:
        video.write(cv2.imread(join(aux_dir, img)))

    cv2.destroyAllWindows()
    video.release()


def natural_sort(images_list):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(images_list, key=alphanum_key)


def make_dir(dirName):
    try:
        mkdir(dirName)
    except OSError:
        pass


def main(parser, args):
    global VIDEO
    if args.video:
        VIDEO = args.video
        
    if args.segm:
        print ("Entrenando las marcas correspondiente...")
        mark_train(args)
        print ("Empezando analisis del video...")
        clf = training()
        analysis(clf, args, segm=True)
    elif args.analy:
        print ("Entrenando las marcas correspondientes...")
        mark_train(args)
        print ("Empezando analisis del video...")
        clf = training()
        analysis(clf, args)

    if args.genVideo:
        gen_video(args.output, args.genVideo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='robotica.py')
    
    parser.add_argument('-v', '--video',
                        help='Select a different video.')


    parser.add_argument('-o', '--output',
                        default='video_output',
                        help='Choose the output video name.')



    group = parser.add_argument_group('Commands')

    group.add_argument('-s', '--segm',
                       action='store_true',
                       help='Start segmentation process.')

    group.add_argument('-a', '--analy',
                       action='store_true', default='True',
                       help='Start analysis process.')

    group.add_argument('-g', '--genVideo',
                       choices=['segm', 'norm', 'analy', 'chull'],
                       nargs='?', const='analy',
                       help='Generate choosen procedure video.')

    args = parser.parse_args()

    main(parser, args)
