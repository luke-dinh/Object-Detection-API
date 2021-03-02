import numpy as np

class BoundingBox:

    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.scores = -1 

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

def sigmoid(x):
    return 1./(1 + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = sigmoid(netout[..., :2])
    netout[..., 4:]  = sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
 
    for i in range(grid_h*grid_w):
	    row = i / grid_w
	    col = i % grid_w
	    for b in range(nb_box):
			# 4th element is objectness score
		    objectness = netout[int(row)][int(col)][b][4]
		    if(objectness.all() <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
		    x, y, w, h = netout[int(row)][int(col)][b][:4]
		    x = (col + x) / grid_w # center position, unit: image width
		    y = (row + y) / grid_h # center position, unit: image height
		    w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
		    h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
		    classes = netout[int(row)][col][b][5:]
		    box = BoundingBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
		    boxes.append(box)
    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3

def bbox_iou(bbox1, bbox2):
    intersect_1 = interval_overlap([bbox1.xmin, bbox1.xmax], [bbox2.xmin, bbox2.xmax])
    intersect_2 = interval_overlap([bbox1.ymin, bbox1.ymax], [bbox2.ymin, bbox2.ymax])
    intersect = intersect_1 * intersect_2
    w1, h1 = bbox1.xmax - bbox1.xmin, bbox1.ymax - bbox1.ymin
    w2, h2 = bbox2.xmax - bbox2.xmin, bbox2.ymax - bbox2.ymin

    union = w1*h1 + w2*h2 - intersect
    return float(intersect)/union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
	    nb_class = len(boxes[0].classes)
    else:
	    return
    for c in range(nb_class):
	    sorted_indices = np.argsort([-box.classes[c] for box in boxes])
	    for i in range(len(sorted_indices)):
		    index_i = sorted_indices[i]
		    if boxes[index_i].classes[c] == 0: continue
		    for j in range(i+1, len(sorted_indices)):
			    index_j = sorted_indices[j]
			    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
				    boxes[index_j].classes[c] = 0
