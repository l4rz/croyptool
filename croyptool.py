import numpy as np
import tensorflow as tf
import cv2 as cv
import os, glob, time
import random, sys

'''

usage: croyptool.py <source images dir> <cropped images dir>

    space bar: accept suggested cropping, save cropped image
    esc: reject and continue
    q: reject and quit croytool

for squaring use imagick convert:

for i in *.jpg; do convert $i  -resize 1024x1024 ../1/$i; done
for i in *.jpg; do convert $i  -gravity center -background white -extent 1024x1024 ../2/$i; done

'''

# Read the graph.
with tf.gfile.FastGFile('frozen_inference_graph_faces.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    filelist = sorted(glob.glob(sys.argv[1]+'/*.jpg'))

    for file in filelist:

        img = cv.imread(file)
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        print("num_detections",num_detections)
        #rimg = cv.resize(img,(int(cols/2),int(rows/2)))
        rimg = cv.imread(file)
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.5:
                print("good score:",score,"for file",file)
                x = int(bbox[1] * cols )
                y = int(bbox[0] * rows   )
                right = int(bbox[3] * cols ) 
                bottom = int(bbox[2] * rows )

                # check whether face is vertical
                f_width = right - x
                f_height = bottom - y
                print ("aspect ratio is 1 to", f_height / f_width)

                if f_height / f_width > 1 and f_height / f_width < 1.6:
                    # adjust these parameters to modify dimensions of crop box
                    crop_x = int(x - f_width * 1.0) # 1.5 for moar squared
                    crop_y = int(y - f_height * 0.5)
                    crop_right = int(x + f_width * 2.0) # 2.5 for moar squared
                    crop_bottom = int(y + f_width * 3.5) # was 4

                    rimg = cv.rectangle(  rimg,   (x, y), (right, bottom), (125, 255, 51), thickness=2)
                    rimg = cv.rectangle(  rimg,   (crop_x, crop_y), (crop_right, crop_bottom), (255, 255, 0), thickness=2)

                    cv.imshow('croyptool', rimg)
                    cv.moveWindow('croyptool', 0, 0)
                    k = cv.waitKey()
                    if k == ord('q'):
                        sys.exit()
                    elif k == 27:         
                        print(file, 'skipped')
                        #print()
                        break
                    elif k == 32:        
                        if crop_x < 0:
                            crop_x = 0
                        if crop_y < 0:
                            crop_y = 0
                        print('saving to',sys.argv[2] + '/' + os.path.basename(file), ":", crop_y,crop_bottom,crop_x,crop_right)
                        crop_img = img[crop_y:crop_bottom, crop_x:crop_right]
                        cv.imwrite(sys.argv[2] + '/cropped' + os.path.basename(file), crop_img)

  



