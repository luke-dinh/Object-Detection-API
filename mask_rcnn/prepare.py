from os import listdir
import re
from xml.etree import ElementTree
from numpy import zeros, asarray
from Mask_RCNN.mrcnn.utils import Dataset, extract_bboxes
from Mask_RCNN.mrcnn.visualize import display_instances
from matplotlib import pyplot

class kangaroo_dataset(Dataset):

    def load_dataset(self, datadir, is_train=True):

        #define one class
        self.add_class("dataset", 1, "kangaroo")

        #dataset location
        images_dir = datadir + "/images/"
        annots_dir = datadir + "/annots/"

        #find all images
        for filename in listdir(images_dir):

            #get image id
            image_id = filename[:-4]

            #skip bad images
            if image_id in ['00090']:
                continue

            #build train set with 150 images:

            if is_train and int(image_id) >= 150:
                continue

            #build val set with 33 images

            if not is_train and int(image_id) < 150:
                continue

            img_path = images_dir + filename
            annot_path = annots_dir + image_id + '.xml'

            #add to dataset

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=annot_path)

    def extract_boxes(self, filename):

        #load and parse the file

        tree = ElementTree.parse(filename)
        
        #get the root of the document

        root = tree.getroot()

        #extract each bounding box

        bbox = []

        for box in root.findall('.//bndbox'):

            xmin = int(box.find('xmin').text)
            xmax = int(box.find('xmax').text)
            ymin = int(box.find('ymin').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            bbox.append(coors)

        #extract image dimensions

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)

        return bbox, width, height

    def load_mask(self, image_id):

        #get details of images
        
        info = self.image_info[image_id]

        #get box fike location

        path = info['annotation']

        #load xml file

        boxes, w, h = self.extract_boxes(path)

        #convert mask to np.array

        masks = zeros([h, w, len(boxes)], dtype='uint8')

        #create masks

        class_ids = []

        for i in range(len(boxes)):

            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('kangaroo'))
        return masks, asarray(class_ids, dtype='int32')

    #load an image reference

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# For testing the dataset 

# # load an image
# image_id = 0
# image = train_set.load_image(image_id)
# print(image.shape)
# # load image mask
# mask, class_ids = train_set.load_mask(image_id)
# print(mask.shape)
# # plot image
# pyplot.imshow(image)
# # plot mask
# pyplot.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
# pyplot.show()

# image_id = 1
# image = train_set.load_image(image_id=image_id)
# mask, class_ids = train_set.load_mask(image_id)
# bbox = extract_bboxes(mask)
# display_instances(image, bbox, mask, class_ids, train_set.class_names)