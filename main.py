import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
tf.compat.v1.disable_eager_execution()

from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

import yolo_utils

def yolo_filter_boxes(box_confidence , boxes, box_class_probs, threshold = 0.6):
    """
    Filter the confidence level of objects and classifications by thresholds.

    Parameters：
        box_confidence  - tensor, dimension(19,19,5,1), containing all the anchor boxes in the 5 anchor boxes predicted by each cell in 19x19 cells (confidence probability of some objects).
        boxes - tensor, dimension(19,19,5,4), containing the (px,py,ph,pw) of all anchor boxes。
        box_class_probs - tensor, dimension(19,19,5,80), containing the detection rates of all objects (c1,c2,c3，···，c80) in all anchors in all cells.
        threshold - real number, threshold. If the probability of a classification prediction is higher than it, then the probability of the classification prediction is preserved.

    Returns：
        scores - tensor, dimension(None,), containing the classification rates of all saved anchor boxes.
        boxes - tensor, dimension(None,4), containing the (b_x, b_y, b_h, b_w) of all saved anchor boxes.
        classess - tensor, dimension(None,), containing the indexs of all saved anchor boxes.

    Note："None" is because that the exact number of boxes is unknown, as it is decided by the thresholds.
          e.g., if there are 10 anchor boxes, the actual size of the output of score will be (10,).
    """

    # Step1: Calculate the scores of anchor boxes.
    box_scores  = box_confidence * box_class_probs

    # Step2: Find the index of the anchor box with maximum, and the according scores.第二步：找到最大值的锚框的索引以及对应的最大值的锚框的分数
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)

    # Step3: Create a filtering mask according to the threshold.
    filtering_mask = (box_class_scores >= threshold)

    # Step4: Use the mask onto scores, boxes and classes.
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)

    return scores , boxes , classes
'''
with tf.Session() as test_a:
    box_confidence = tf.random_normal([19,19,5,1], mean=1, stddev=4, seed=1)
    boxes = tf.random_normal([19,19,5,4],  mean=1, stddev=4, seed=1)
    box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)

    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))

    test_a.close()
'''
def iou(box1, box2):
    """
    Compute the IOU of two anchor boxes.

    Parameters：
        box1 - The first anchor box, tuple, (x1, y1, x2, y2)
        box2 - The second anchor box, tuple, (x1, y1, x2, y2)

    Return：
        iou - Real number, Intersection over Union.
    """
    # Compute the area of intersection.
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = (xi1-xi2)*(yi1-yi2)

    # Compute the union using the equation: Union(A,B) = A + B - Inter(A,B).
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area

    # Compute the IOU.
    iou = inter_area / union_area

    return iou
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Non-max suppression (NMS) for anchor boxes.

    Parameters：
        scores - tensor, dimension(None,), the output of yolo_filter_boxes().
        boxes - tensor, dimension(None,4), the output of yolo_filter_boxes(), scaled to the size of images.
        classes - tensor, dimension(None,), the output of yolo_filter_boxes().
        max_boxes - ints, the maximum of the number of predicted anchors.
        iou_threshold - Real number, the threshold of IOU.

    Returns：
        scores - tensor, dimension(,None), the possible value of the prediction for every anchor box.
        boxes - tensor, dimension(4,None), the coordinate of predicted anchor boxe.
        classes - tensor, dimension(,None), the predicted class for every anchor box.

    Note："None" is obviously smaller than the value of max_boxes. This function will also change the dimensions of scores, boxes, classes, providing convenience for the next step.

    """
    max_boxes_tensor = K.variable(max_boxes,dtype="int32") # used for tf.image.non_max_suppression()
    K.get_session().run(tf.compat.v1.variables_initializer([max_boxes_tensor])) # innitialising max_boxes_tensor

    # use tf.image.non_max_suppression() to botain the inde list of saved anchor boxes.
    nms_indices = tf.image.non_max_suppression(boxes, scores,max_boxes,iou_threshold)

    # useK.gather()to opt the saved anchor boxes.
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes
'''
with tf.Session() as test_b:
    scores = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed = 1)
    classes = tf.random_normal([54,], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))

    test_b.close()
'''
def yolo_eval(yolo_outputs, image_shape=(720.,1280.), 
              max_boxes=10, score_threshold=0.6,iou_threshold=0.5):
    """
    Convert the initial output of YOLO (many boxes) to the prediction boxs and their scores, coordinates and classes.

    Parameters：
        yolo_outputs - The output of the model(for an image whose dimension is(608,608,3), containing 4 tensor variables.
                        box_confidence ： tensor, dimension(None, 19, 19, 5, 1)
                        box_xy         ： tensor, dimension(None, 19, 19, 5, 2)
                        box_wh         ： tensor, dimension(None, 19, 19, 5, 2)
                        box_class_probs： tensor, dimension(None, 19, 19, 5, 80)
        image_shape - tensor, dimension(2,), including the dimension of inut image, here (608.,608.).
        max_boxes - ints, the maximun of the number of predicted anchor boxes.
        score_threshold - real number, possible threshold.
        iou_threshold - real number, the threshold of IOU.

    返回：
        scores - tensor, dimension(,None), the possible value for the prediction of every anchor box.
        boxes - tensor, dimension(4,None), the coordinate of predicted anchor box.
        classes - tensor, dimension(,None), the predicted class of every anchor box.
    """

    # obtain the output of YOLO model
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # conver the centre to corner
    boxes = yolo_boxes_to_corners(box_xy,box_wh)

    # filter the confidence
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # scale the anchor box to adapt to the original image
    boxes = yolo_utils.scale_boxes(boxes, image_shape)

    # NMS
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes
'''
with tf.Session() as test_c:
    yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)

    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))

    test_c.close()
'''
sess = K.get_session()
class_names = yolo_utils.read_classes("model_data/coco_classes.txt")
anchors = yolo_utils.read_anchors("model_data/yolo_anchors.txt")
image_shape = (720.,1280.)
yolo_model = load_model("model_data/yolov2.h5")
#yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
def predict(sess, image_file, is_show_info=True, is_plot=True):
    """
    Run the computation stored in sess to predict the border box for image_file, print the predicted image and related information.

    Parameters：
        sess - the TensorFlow/Keras session including the YOLO computation.
        image_file - names of images stored in the images folder.
    Returns：
        out_scores - tensor, dimension(None,), the possible value for the prediction of anchor box.
        out_boxes - tensor, dimension(None,4), containing the location of predicted box.
        out_classes - tensor类, dimension(None,), the classification index of predicted box.
    """
    # pre-processing of images
    image, image_data = yolo_utils.preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # run the session
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict = {yolo_model.input:image_data, K.learning_phase(): 0})

    # print prediction
    if is_show_info:
        print("在" + str(image_file) + "中找到了" + str(len(out_boxes)) + "个锚框。")

    # the colour to draw the boder box
    colors = yolo_utils.generate_colors(class_names)

    # draw the border boxes in images
    yolo_utils.draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    # save the drawn images
    image.save(os.path.join("out", image_file), quality=100)

    # print the images with border boxes drawn
    if is_plot:
        output_image = scipy.misc.imread(os.path.join("out", image_file))
        plt.imshow(output_image)

    return out_scores, out_boxes, out_classes

for i in range(1,121):

    # zeros to be added in the front
    num_fill = int( len("0000") - len(str(1))) + 1
    # index
    filename = str(i).zfill(num_fill) + ".jpg"
    print("current file：" + str(filename))

    # start drawing with no print
    out_scores, out_boxes, out_classes = predict(sess, filename,is_show_info=False,is_plot=False)

print("Drawing completed!")
