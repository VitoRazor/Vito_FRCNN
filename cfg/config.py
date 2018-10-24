from keras import backend as K

class Config:
    def __init__(self):
        self.verbose = True

        self.network = 'resnet50'

        # setting for data augmentation
        self.use_horizontal_flips = True
        self.use_vertical_flips = True
        self.rot_90 = True
        self.num_anchors=9
        # anchor box scales
        self.anchor_box_scales = [128, 256, 512]

        # anchor box ratios
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]

        # size to resize the smallest side of the image
        self.im_size = 600

        # image channel-wise mean to subtract
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        # number of ROIs at once
        self.num_rois = 32

        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = 16

        self.balanced_classes = False

        # scaling the stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        # placeholder for the class mapping, automatically generated by the parser
        self.class_mapping = 10
        self.num_classes=10

        # location of pretrained weights for the base network
        # weight files can be found at:
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

        self.model_path = 'saved_model/kitti_frcnn_last.hdf5'

        # params add by me
        self.data_dir = '../'
        self.num_epochs = 3000

        self.kitti_simple_label_file = 'kitti_simple_label.txt'

        # TODO: this field is set to simple_label txt, which in very simple format like:
        # TODO: /path/image_2/000000.png,712.40,143.00,810.73,307.92,Pedestrian, see kitti_simple_label.txt for detail
        self.simple_label_file = 'simple_label.txt'

        self.config_save_file = 'config.pickle'