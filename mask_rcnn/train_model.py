from mask_rcnn.prepare import kangaroo_dataset
from numpy import expand_dims, mean
from mask_rcnn.Mask_RCNN.mrcnn.model import MaskRCNN, mold_image, load_image_gt
from mask_rcnn.Mask_RCNN.mrcnn.config import Config

class Kangaroo_Config(Config):

    #define the name of the configuration

    NAME = 'kangaroo_cfg'

    #Number of classes (kangaroo + background)

    NUM_CLASSES = 1 + 1

    #Step per epoch

    STEPS_PER_EPOCH = 131

# train set
train_set = kangaroo_dataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()

# test/val set
test_set = kangaroo_dataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()

#prepare config
config = Kangaroo_Config()
config.display()

#define model
model = MaskRCNN(mode='training', model_dir='./', config=config)

#load weight
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')