from utils import correct_yolo_boxes, decode_netout, do_nms, BoundingBox
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

def load_image(img_file, shape):
    #get image shape
    image = load_img(img_file)
    width, height = image.size
    #reshape image 
    image = load_img(img_file, target_size=shape)
    image = img_to_array(image)
    #scale image
    image /= 255.0
    image =  np.expand_dims(image, 0)
    return image, width, height

def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = [], [], []
    #enumerate all boxes
    for box in boxes:
        #enumerate all labels
        for i in range(len(labels)):
            #check if threshold of the box is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
    
    return v_boxes, v_labels, v_scores

def draw_boxes(img_file, v_boxes, v_labels, v_scores):
    data = plt.imread(img_file)
    plt.imshow(data)
    ax = plt.gca()
    #plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        #get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        #caculate width, height
        width, height = x2 - x1, y2 - y1
        #create shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='green')
        #draw the boxes
        ax.add_patch(rect)
        #draw text and scores
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        plt.text(x1, y1, label, color='white')
    plt.show()

model = load_model('yolo/model.h5')
input_w, input_h = 416, 416
img_file = 'yolo/zebra.jpg'
image, image_w, image_h = load_image(img_file, (input_w, input_h))
yhat = model.predict(image)
print([a.shape for a in yhat])
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
# define the probability threshold for detected objects
class_threshold = 0.6
boxes = list()

for i in range(len(yhat)):
    boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

do_nms(boxes, 0.5)

labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# get the details of the detected objects
v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
# summarize what we found
for i in range(len(v_boxes)):
    print(v_labels[i], v_scores[i])
# draw what we found
draw_boxes(img_file, v_boxes, v_labels, v_scores)