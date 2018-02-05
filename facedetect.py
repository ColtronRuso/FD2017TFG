
import numpy as np
import cv2
import cv2.cv as cv
#from video import create_capture
#from common import clock, draw_str
import sys
import os
import glob
import argparse
import time

import llibreries as lib


# Argumentos de entrada
parser = argparse.ArgumentParser(description='Runs face detection on a folder of images.')
parser.add_argument('-input_path', type=str,
                   help='path where the input images are stored',
                   default = None)
parser.add_argument('-output_path', type=str,
                   help='path where the result is stored',
                   default='./out')


parser.add_argument('-factor_scale', type=float, help="Escala del factor de busqueda",
                default=1.1)
parser.add_argument('-min_neighbors', type=int, help="Veins minimos",
                default=2)
parser.add_argument('-min_size', type=int, help="Tamany minim en pixels de la cara",
                default=-1)
parser.add_argument('-max_size', type=int, help="Tamany maxim en pixels de la cara",
                default=-1)

parser.add_argument('-haarcascades', type=str, help="fitxers xml per a filtre haar",
                #default=['haarcascades/haarcascade_frontalface_alt2.xml', 'haarcascades/haarcascade_profileface.xml','haarcascade_frontalface_alt.xml','haarcascade_frontalcatface.xml'])
                default=['haarcascades/haarcascade_frontalface_alt2.xml', 'haarcascades/haarcascade_profileface.xml'])
parser.add_argument('-cascade', type=str, help="Per utilitzar un sol fitxer xml de filtre",
                default='./haarcascades/haarcascade_frontalface_alt.xml')

parser.add_argument('-option', type=str, help="Opcio desitjada a realitzar",
                default='video')
parser.add_argument('-image', type=str, help="directori i nom del fitxe a llegir")

parser.add_argument('-extensions', type=str, help="extensions que es volen comprovar",
                default=['*.jpg','*.jpeg','*.png'])
parser.add_argument('-type_vector', type=str, help="Tipus de vector que utilitzara per agrupar i classificar",
                default='face') # Tipus -> ('face','face-clothes')

parser.add_argument('-level_blur', type=float, help="Nivel de distorsion de la cara entre 0.0 a 1.0",
                default=0.3)

parser.add_argument('-aviso', type=str, help="Per mostrar avisos per pantalla",
                default=False)

parser.add_argument('-source_video', type=int, help="Definir l'origen del video",
                default=0)


argos = parser.parse_args()


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    
    if argos.option in ('video','blur_image','blur_dir'):
        if argos.option == 'video':

            #args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=','level_blur='])
            
            try: video_src = argos.source_video[0]
            except: video_src = 0
            
            #args = dict(args)
            cascade_fn = argos.cascade
            #cascade_fn = args.get('--cascade', "./haarcascades/haarcascade_frontalface_alt.xml")

            cascade = cv2.CascadeClassifier(cascade_fn)

            cam = lib.vd.create_capture(video_src, fallback='synth:bg=./images/lena.jpg:noise=0.05')

            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                
                #Mesures de distorsio
                w,h = gray.shape
                if argos.max_size == -1:
                    size_max = int(max([w*0.35,h*0.35]))
                else:
                    size_max = min([w,h])
                    
                if argos.min_size == -1:
                    size_min = int(min([w*0.05,h*0.05]))
                else:
                    size_min = argos.min_size
                
                t = lib.clock()
                # Localitzacio imatge
                rects = lib.findFaces.detect(gray, cascade, argos, size_min, size_max)
                vis = img.copy()
                if argos.aviso:
                    lib.findFaces.draw_rects(vis, rects, (0, 255, 0))
                for x1, y1, x2, y2 in rects:
                    if argos.level_blur > 0.01:
                        roi = vis[y1:y2, x1:x2]
                        # Distorsionar cara se multiplica longuitud cuadrado cara por nivel
                        img_blur = cv2.blur(roi,(int(y2*argos.level_blur),int(y2*argos.level_blur)))
                        vis[y1:y2, x1:x2] = img_blur
                    
                dt = lib.clock() - t

                lib.draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
                cv2.imshow('facedetect', vis)

                if 0xFF & cv2.waitKey(5) == 27:
                    break
            cv2.destroyAllWindows()

        if argos.option == 'blur_image':
            if argos.image is not None:
                lib.findFaces.imageSmooth(argos.image,argos.output_path, blur_lvl=argos.level_blur,scale_factor=argos.factor_scale,
                                          min_size=argos.min_size, max_size=argos.max_size,
                                          face_cascade_paths=argos.haarcascades, 
                                          exts=argos.extensions,
                                          min_neighbors=argos.min_neighbors)
            else:
                print "Falta argument -image << .../directori/imatge.jpg >>"
            
        if argos.option == 'blur_dir':
            if argos.input_path is not None:
                start = time.time()
                lib.findFaces.smoothFaces(argos.input_path,argos.output_path,blur_lvl = argos.level_blur,scale_factor = argos.factor_scale,
                                        face_cascade_paths = argos.haarcascades, min_size = argos.min_size, max_size = argos.max_size,
                                        exts=argos.extensions,min_neighbors=argos.min_neighbors)
                
                if argos.aviso:
                        print("Temps transcurregut per distorsionar les cares: {} seconds.".format(time.time() - start))
                
                images =[]
                for ext in argos.extensions:    
                    images.extend(glob.glob(argos.output_path+'/'+ext))
                
                num = 0
                for nfaces in images:
                    num += int(nfaces.split('/')[-1].split('.')[0].split('-')[-1])
                
                if argos.aviso:
                    print "S'han distorsionat " + str(num) + " Cara(es)"
            else:
                print "Falta argument -input_path << .../directori>>"
    else:
        
        print 'Opcio inexistent ' + argos.option 
