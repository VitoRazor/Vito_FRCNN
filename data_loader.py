import numpy as np 
import scipy
from glob import glob
import random
import pickle
import cv2
import threading
import copy
import itertools
path='../datasets'


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height)

def union(au, bu, area_intersection):
	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection
	return area_union


def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h


def iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height


class SampleSelector:
	def __init__(self, class_count):
		# ignore classes that have zero samples
		# find all class
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		#set iter cycle 
		self.class_cycle = itertools.cycle(self.classes)
		self.curr_class = next(self.class_cycle)

	def skip_sample_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height):

	downscale = float(C.rpn_stride)
	anchor_sizes = C.anchor_box_scales
	anchor_ratios = C.anchor_box_ratios
	num_anchors = len(anchor_sizes) * len(anchor_ratios)	

	# calculate the output map size based on the network architecture

	(output_width, output_height) = get_img_output_length(resized_width, resized_height)

	n_anchratios = len(anchor_ratios)
	
	# initialise empty output objectives
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

	num_bboxes = len(img_data['bboxes'])

	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
	best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

	# get the GT box coordinates, and resize to account for image resizing
	gta = np.zeros((num_bboxes, 4))
	for bbox_num, bbox in enumerate(img_data['bboxes']):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
		gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
		gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
		gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
	
	# rpn ground truth

	for anchor_size_idx in range(len(anchor_sizes)):
		for anchor_ratio_idx in range(n_anchratios):
			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]	
			
			for ix in range(output_width):					
				# x-coordinates of the current anchor box	
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
				
				# ignore boxes that go across image boundaries					
				if x1_anc < 0 or x2_anc > resized_width:
					continue
					
				for jy in range(output_height):

					# y-coordinates of the current anchor box
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					# ignore boxes that go across image boundaries
					if y1_anc < 0 or y2_anc > resized_height:
						continue

					# bbox_type indicates whether an anchor should be a target 
					bbox_type = 'neg'

					# this is the best IOU for the (x,y) coord and the current anchor
					# note that this is different from the best IOU for a GT bbox
					best_iou_for_loc = 0.0

					for bbox_num in range(num_bboxes):
						
						# get IOU of the current GT box and the current anchor box
						curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])
						# calculate the regression targets if they will be needed
						if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
							cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
							cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
							cxa = (x1_anc + x2_anc)/2.0
							cya = (y1_anc + y2_anc)/2.0

							tx = (cx - cxa) / (x2_anc - x1_anc)
							ty = (cy - cya) / (y2_anc - y1_anc)
							tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
							th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
						
						if img_data['bboxes'][bbox_num]['class'] != 'bg':

							# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
							if curr_iou > best_iou_for_bbox[bbox_num]:
								best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
								best_iou_for_bbox[bbox_num] = curr_iou
								best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
								best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]

							# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
							if curr_iou > C.rpn_max_overlap:
								bbox_type = 'pos'
								num_anchors_for_bbox[bbox_num] += 1
								# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
								if curr_iou > best_iou_for_loc:
									best_iou_for_loc = curr_iou
									best_regr = (tx, ty, tw, th)

							# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
							if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
								# gray zone between neg and pos
								if bbox_type != 'pos':
									bbox_type = 'neutral'

					# turn on or off outputs depending on IOUs
					if bbox_type == 'neg':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'neutral':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
					elif bbox_type == 'pos':
						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
						start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
						y_rpn_regr[jy, ix, start:start+4] = best_regr

	# we ensure that every bbox has at least one positive RPN region

	for idx in range(num_anchors_for_bbox.shape[0]):
		if num_anchors_for_bbox[idx] == 0:
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1:
				continue
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
				best_anchor_for_bbox[idx,3]] = 1
			start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

	num_pos = len(pos_locs[0])

	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions. We also limit it to 256 regions.
	num_regions = 256

	if len(pos_locs[0]) > num_regions/2:
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])

	if augment:
		rows, cols = img.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2        
				elif angle == 0:
					pass

	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]
	return img_data_aug, img
class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)	
class DataLoader():
    def __init__(self, cfg, img_res=(64,64,1)):
        self.cfg=cfg
        self.cfg.simple_label_file = 'kitti_simple_label.txt'
        
        self.img_res=img_res
        self.all_images, self.classes_count, self.class_mapping = self.get_data(self.cfg.simple_label_file)
        if 'bg' not in self.classes_count:
            self.classes_count['bg'] = 0
            self.class_mapping['bg'] = len(self.class_mapping)

        self.cfg.class_mapping = self.class_mapping
        inv_map = {v: k for k, v in self.class_mapping.items()}
        print('Training images per class:')
        print(self.classes_count)
        print('Num classes (including bg) = {}'.format(len(self.classes_count)))
        random.shuffle(self.all_images)
        num_imgs = len(self.all_images)
        train_imgs = [s for s in self.all_images if s['imageset'] == 'trainval']
        val_imgs = [s for s in self.all_images if s['imageset'] == 'test']

        print('Num train samples {}'.format(len(train_imgs)))
        print('Num val samples {}'.format(len(val_imgs)))

    def load_data(self, batch_size=1, is_testing=False):
        # data_type = "train" if not is_testing else "val"
        # #glob.glob()-----list all file paths
        # path = glob(self.path+'/%s/%s/*' % (self.dataset_name, data_type))
        # batch = np.random.choice(path, size=batch_size)

        # imgs_A=[]
        # for img in batch:
        #     img=self.imread(img)
        #     img_A=cv2.resize(img,self.img_res,interpolation=cv2.INTER_CUBIC)  
            
        #     if not is_testing and np.random.random() > 0.5:
        #             img_A = np.fliplr(img_A)
        #     imgs_A.append(img_A)
        return 0

    def load_batch(self, batch_size=1, is_testing=False):
        # data_type = "train" if not is_testing else "val"
        # path = glob(self.path+'/%s/%s/*' % (self.dataset_name, data_type))

        # # shuffle the datasets

        # random.shuffle(path)
        # self.n_batches=int(len(path)/batch_size)
        # while True:
        #     for i in range(self.n_batches-1):
        #         batch = path[i*batch_size:(i+1)*batch_size]
        #         imgs_A = []
        #         for img in batch:
        #             img=self.imread(img)
        
        #             img_A=cv2.resize(img, self.img_res[0:2], interpolation=cv2.INTER_CUBIC) 
        #             if not (self.img_res[2] > 1):
        #                 img_A=np.expand_dims(img_A,axis=2)
        #             if not is_testing and np.random.random() > 0.5:
        #                     img_A = np.fliplr(img_A)
        #             imgs_A.append(img_A)
        #         imgs_A= np.array(imgs_A)/127.5-1.
        #         yield imgs_A
        return 0

    def get_anchor_gt(self, mode='train'):

        # The following line is not useful with Python 3.5, it is kept for the legacy
        # all_img_data = sorted(all_img_data)
        all_img_data=self.all_images
        class_count=self.classes_count
        C = self.cfg
        sample_selector = SampleSelector(class_count)
        

        while True:
            if mode == 'train':
                random.shuffle(all_img_data)

            for img_data in all_img_data:
                try:

                    if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                        continue

                    # read in image, and optionally add augmentation

                    if mode == 'train':
                        img_data_aug, x_img = augment(img_data, C, augment=False)
                    else:
                        img_data_aug, x_img = augment(img_data, C, augment=False)

                    (width, height) = (img_data_aug['width'], img_data_aug['height'])
                    (rows, cols, _) = x_img.shape

                    assert cols == width
                    assert rows == height

                    # get image dimensions for resizing
                    (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

                    # resize the image so that smalles side is length = 600px
                    x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                    try:
                        y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height)
                    except:
                        continue

                    # Zero-center by mean pixel, and preprocess image

                    x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
                    x_img = x_img.astype(np.float32)
                    x_img[:, :, 0] -= C.img_channel_mean[0]
                    x_img[:, :, 1] -= C.img_channel_mean[1]
                    x_img[:, :, 2] -= C.img_channel_mean[2]
                    x_img /= C.img_scaling_factor

                    x_img = np.transpose(x_img, (2, 0, 1))
                    x_img = np.expand_dims(x_img, axis=0)

                    y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

                    x_img = np.transpose(x_img, (0, 2, 3, 1))
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                    yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

                except Exception as e:
                    print(e)
                    continue

    def load_img(self, path):
        img = self.imread(path)
        img_A=cv2.resize(img,self.img_res,interpolation=cv2.INTER_CUBIC)  

        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        if (self.img_res[2] > 1):
            return cv2.imread(path).astype(np.float)
        else:
            a=cv2.imread(path,0).astype(np.float)
            return np.expand_dims(a,axis=2)

    def get_data(self,input_path):
        found_bg = False
        all_imgs = {}
        classes_count = {}
        class_mapping = {}
        visualise = True
        with open(input_path, 'r') as f:

            print('Parsing annotation files')

            for line in f:
                line_split = line.strip().split(',')
                (filename, x1, y1, x2, y2, class_name) = line_split
                filename ="../"+filename
                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if class_name not in class_mapping:
                    if class_name == 'bg' and not found_bg:
                        print('Found class name with special name bg. Will be treated as a'
                            ' background region (this is usually for hard negative mining).')
                        found_bg = True
                    class_mapping[class_name] = len(class_mapping)

                if filename not in all_imgs:
                    all_imgs[filename] = {}

                    img = cv2.imread(filename)
                    (rows, cols) = img.shape[:2]
                    all_imgs[filename]['filepath'] = filename
                    all_imgs[filename]['width'] = cols
                    all_imgs[filename]['height'] = rows
                    all_imgs[filename]['bboxes'] = []
                    if np.random.randint(0, 6) > 0:
                        all_imgs[filename]['imageset'] = 'trainval'
                    else:
                        all_imgs[filename]['imageset'] = 'test'

                all_imgs[filename]['bboxes'].append(
                    {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)),
                    'y2': int(float(y2))})

            all_data = []
            for key in all_imgs:
                all_data.append(all_imgs[key])

            # make sure the bg class is last in the list
            if found_bg:
                if class_mapping['bg'] != len(class_mapping) - 1:
                    key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                    val_to_switch = class_mapping['bg']
                    class_mapping['bg'] = len(class_mapping) - 1
                    class_mapping[key_to_switch] = val_to_switch

            return all_data, classes_count, class_mapping
