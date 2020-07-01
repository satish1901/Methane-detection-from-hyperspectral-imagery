# USAGE
# python boats.py --mode train
# python boats.py --mode investigate
# python boats.py --mode predict --image examples/boats_01.jpg
# python boats.py --mode predict --image examples/boats_01.jpg \
# 	--weights logs/boats20181018T0624/mask_rcnn_boats_0015.h5

# import the necessary boats
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import utils
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import json
import cv2
import os

import logging
import coloredlogs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train-plumes')
coloredlogs.install(level='DEBUG', logger=logger)


# initialize the class names dictionary
#CLASS_NAMES = {1: "point_source", 2: "diffused_source", 3: "no_methane"}
CLASS_NAMES = {1: "point_source"}

# initialize the path to the Mask R-CNN pre-trained on COCO
COCO_PATH = "./mask_rcnn_coco.h5"

# initialize the name of the directory where logs and output model
# snapshots will be stored
LOGS_AND_MODEL_DIR = "logs"

def init_configs(TRAIN_DIR, train_dir, ANNOT_PATH):
	# initialize the dataset path, images path, and annotations file path
	ALL_IMAGES_PATH  = os.path.sep.join([TRAIN_DIR, train_dir])
	print(ALL_IMAGES_PATH)

	# initialize the amount of data to use for training
	TRAINING_SPLIT = 0.66

	# grab all image paths, then randomly select indexes for both training
	# and validation
	ALL_IMAGE_PATHS = sorted(list(paths.list_files(ALL_IMAGES_PATH)))
	print(len(ALL_IMAGE_PATHS))
	#print(ANNOT_PATH)

	# this keeps the list of files which all have the objects in them
	files_with_roi = []
	IMAGE_PATHS = []
	# pick on those images path which are in the annotation file
	annot_file = json.loads(open(ANNOT_PATH).read())
	for (ignr, annot_filename) in sorted(annot_file.items()):
		files_with_roi.append(annot_filename["filename"])
	
	# now create a new list of imagepaths the of the particular files
	# available with annotations
	for fileIdx in range(0,len(ALL_IMAGE_PATHS)):
		file_path = ALL_IMAGE_PATHS[fileIdx]
		file_name = file_path.split(os.path.sep)[-1]
		if file_name not in files_with_roi:
			continue
		else:
			IMAGE_PATHS.append(file_path)

	print("num of annot", len(files_with_roi), "num of images", len(IMAGE_PATHS))

	idxs = list(range(0, len(IMAGE_PATHS)))
	random.seed(42)
	random.shuffle(idxs)
	i = int(len(idxs) * TRAINING_SPLIT)
	trainIdxs = idxs[:i]
	valIdxs = idxs[i:]

	return IMAGE_PATHS, trainIdxs, valIdxs

class BoatsConfig(Config):
	def __init__(self, trainIdxs):
		super().__init__()
		# give the configuration a recognizable name
		self.NAME = "point_source"

	    # set the number of GPUs to use training along with the number of
	    # images per GPU (which may have to be tuned depending on how
	    # much memory your GPU has)
		self.GPU_COUNT = 1
		self.IMAGES_PER_GPU = 1
		self.BATCH_SIZE = 1

		# number of classes (+1 for the background)
		self.NUM_CLASSES = len(CLASS_NAMES) + 1
		self.trainIdxs = trainIdxs
		# set the number of steps per training epoch
		self.STEPS_PER_EPOCH = len(self.trainIdxs) // (self.GPU_COUNT * self.IMAGES_PER_GPU)
		self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

class BoatsInferenceConfig(BoatsConfig):
	# set the number of GPUs and images per GPU (which may be
	# different values than the ones used for training)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# set the minimum detection confidence (used to prune out false
	# positive detections)
	DETECTION_MIN_CONFIDENCE = 0.9

class BoatsDataset(utils.Dataset):
	def __init__(self, imagePaths, annotPath, classNames, width=1024):
		# call the parent constructor
		super().__init__(self)

		# store the image paths and class names along with the width
		# we'll resize images to
		self.imagePaths = imagePaths
		self.classNames = classNames
		self.width = width
		#self.updatedImagePaths = []

		# load the annotation data
		print("-----------------------BoatsDataset----------------")
		self.annots = self.load_annotation_data(annotPath)
		self.img_cnt = 0

	def load_annotation_data(self, annotPath):
		# load the contents of the annotation JSON file (created
		# using the VIA tool) and initialize the annotations
		# dictionary
		annotations = json.loads(open(annotPath).read())
		annots = {}

		# loop over the file ID and annotations themselves (values)
		for (fileID, data) in sorted(annotations.items()):
			# store the data in the dictionary using the filename as
			# the key
			annots[data["filename"]] = data
			#print("name of file", annots[data["filename"]])
			#files_with_roi.append(data["filename"])
		#print(files_with_roi)	

		# return the annotations dictionary
		return annots

	def load_boats(self, idxs):
		# loop over all class names and add each to the 'point_source'
		# dataset
		for (classID, label) in self.classNames.items():
			self.add_class("point_source", classID, label)

		# loop over the image path indexes
		for i in idxs:
			#print("BOAT ID %i" % i)
			# extract the image filename to serve as the unique
			# image ID
			imagePath = self.imagePaths[i]
			filename = imagePath.split(os.path.sep)[-1]
			
			image = np.load(imagePath)
			if filename == 'ang20150419t163741_rdn_95_2.npy':
				print("Filename :", filename)
				print(image[:,:,0].max())
				print(image[:,:,1].max())
				img1 = np.uint8(image[:,:,1]*255)
				img2 = np.uint8(image[:,:,2]*255)
				cv2.imwrite('img1.png', img1)
				cv2.imwrite('img2.png', img2)
				print(image[:,:,2].max())
			(origH, origW) = image.shape[:2]
			image = imutils.resize(image, width=self.width)
			(newH, newW) = image.shape[:2]
			print(newH, newW, origW, origH)

			# add the image to the dataset
			self.add_image("point_source", image_id=filename,
				width=newW, height=newH,
				orig_width=origW, orig_height=origH,
				path=imagePath)

	def load_image(self, imageID):
		# grab the image path, load it, and convert it from BGR to
		# RGB color channel ordering
		p = self.image_info[imageID]["path"]
		#print("path : ", p)
		image = np.load(p)

		# resize the image, preserving the aspect ratio
		image = imutils.resize(image, width=self.width)

		# return the image
		return image

	def load_mask(self, imageID):
		#print(f"Image ID: {imageID}")
		# grab the image info and then grab the annotation data for
		# the current image based on the unique ID
		info = self.image_info[imageID]
		self.img_cnt += 1
		annot = self.annots[info["id"]]

		# allocate memory for our [height, width, num_instances] array
		# where each "instance" effectively has its own "channel"	
		masks = np.zeros((info["height"], info["width"],
			len(annot["regions"])), dtype="uint8")


		# loop over each of the annotated regions
		for (i, region) in enumerate(annot["regions"]):
			# allocate memory for the region mask
			regionMask = np.zeros((info["orig_height"], info["orig_width"]), dtype="uint8")

			# grab the shape and region attributes
			sa = region["shape_attributes"]
			ra = region["region_attributes"]
			ratio = info["width"] / float(info["orig_width"])

			if sa["name"] == "circle":
				# scale the center (x, y)-coordinates and radius of the
				# circle based on the dimensions of the resized image
				# ratio = info["width"] / float(info["orig_width"])
				cX = int(sa["cx"] * ratio)
				cY = int(sa["cy"] * ratio)
				r = int(sa["r"] * ratio)
				# draw a circular mask for the region and store the mask
				# in the masks array
				cv2.circle(regionMask, (cX, cY), r, 1, -1)
			
			elif sa["name"] == "polygon":
				# scale the x and y ppoint corrdinates based on the resizing
				# of the image
				pts_x = np.array(sa["all_points_x"])
				pts_y = np.array(sa["all_points_y"])

				region_pts_x = pts_x.astype('int32').tolist()
				region_pts_y = pts_y.astype('int32').tolist()
				
				# drawing the mask
				regionMask[region_pts_x,region_pts_y] = 255
				
				# logger.info(f"Scaled Points: {type(pts)} \n{pts}")
				# draw the filledpolygon (i.e., the mask)
				#cv2.fillPoly(regionMask, pts, 255)

			elif sa["name"] == "manual_mask":
				print("do soemthing here to save the mask to the array regionMask")
			
			regionMask = imutils.resize(regionMask, width=info["width"])
			masks[:, :, i] = regionMask
	
		# return the mask array and class IDs, which for this dataset
		# is all 1's
		return (masks.astype("bool"), np.ones((masks.shape[-1],),
			dtype="int32"))

if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--mode", required=True,
		help="either 'train', 'predict', or 'investigate'")
	ap.add_argument("-w", "--weights",
		help="optional path to pretrained weights")
	ap.add_argument("-i", "--image",
		help="optional path to input image to segment")
	args = vars(ap.parse_args())
	save_prediction = False

	# check to see if we are training the Mask R-CNN
	if args["mode"] == "train":
		TRAIN_DIR = "../../data/train_data/"
		ANNOT_PATH = "../gt_jsonfile/annotation_plumes.json"
		for train_dir in os.listdir(TRAIN_DIR):
			if(os.path.isdir(os.path.sep.join([TRAIN_DIR, train_dir]))):
				IMAGE_PATHS, trainIdxs, valIdxs = init_configs(TRAIN_DIR, train_dir, ANNOT_PATH)
			else:
				continue
			
			# load the training dataset
			trainDataset = BoatsDataset(IMAGE_PATHS, ANNOT_PATH,
				CLASS_NAMES)
			trainDataset.load_boats(trainIdxs)
			trainDataset.prepare()

			# load the validation dataset
			valDataset = BoatsDataset(IMAGE_PATHS, ANNOT_PATH,
				CLASS_NAMES)
			valDataset.load_boats(valIdxs)
			valDataset.prepare()

			# initialize the training configuration
			config = BoatsConfig(trainIdxs)
			config.display()

			#create a directory to save model
			logs_path = f'./{LOGS_AND_MODEL_DIR}/{train_dir}'
			if not(os.path.isdir(logs_path)):
				print("creating directory for saving model at", logs_path)
				os.mkdir(logs_path)
			else:
				print(logs_path, "already exist")
			# initialize the model and load the COCO weights so we can
			# perform fine-tuning
			model = modellib.MaskRCNN(mode="training", config=config,
				model_dir=logs_path)
			model.load_weights(COCO_PATH, by_name=True,
				exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
					"mrcnn_bbox", "mrcnn_mask"])

			# train *just* the layer heads
			model.train(trainDataset, valDataset, epochs=2,
				layers="heads", learning_rate=config.LEARNING_RATE)

			# unfreeze the body of the network and train *all* layers
			model.train(trainDataset, valDataset, epochs=2,
				layers="all", learning_rate=config.LEARNING_RATE / 10)

	# check to see if we are predicting using a trained Mask R-CNN
	elif args["mode"] == "predict":
		output_without_perceptron = 0.0
		for trained_log in os.listdir(LOGS_AND_MODEL_DIR):
			if not os.path.isdir(os.path.sep.join([LOGS_AND_MODEL_DIR, trained_log])):
				continue
			# initialize the inference configuration
			config = BoatsInferenceConfig()

			# initialize the Mask R-CNN model for inference
			trained_log_path = f'./{LOGS_AND_MODEL_DIR}/{trained_log}'
			model = modellib.MaskRCNN(mode="inference", config=config,
				model_dir=trained_log_path)

			# load our trained Mask R-CNN
			weights = args["weights"] if args["weights"] \
				else model.find_last()
			model.load_weights(weights, by_name=True)

			# load the input image, convert it from BGR to RGB channel
			# ordering, and resize the image
			# image = cv2.imread(args["image"])
			#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			image = np.load(args["image"])
			h,w,q = image.shape
			print(image.shape)
			blank_image = np.ones((h,w,3), np.uint8)
			image = imutils.resize(image, width=1024)
			blank_image = imutils.resize(blank_image, width=1024)
			print(type(image),image.dtype)

			# perform a forward pass of the network to obtain the results
			r = model.detect([image], verbose=1)[0]
			if r["rois"].shape[0] < 1:
				print("Nothing was detected")
				continue
			else:
				print("Detected" ,r['rois'].shape[0], " plumes.")

			# loop over of the detected object's bounding boxes and
			# masks, drawing each as we go along
			print(r["rois"].shape[0])
			print(np.sum(r["masks"][:,:,0]))
			#print(np.sum(r["masks"][:,:,1]))
			for i in range(0, r["rois"].shape[0]):
				mask = r["masks"][:, :, i]
				# mask[500:600, 500:800] = 1
				image = visualize.apply_mask(blank_image, mask,
					(1.0, 0.0, 0.0), alpha=0.5)
				image = visualize.draw_box(blank_image, r["rois"][i],
					(1.0, 0.0, 0.0))

			# convert the image back to BGR so we can use OpenCV's
			# drawing functions
			# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

			# resize the image so it more easily fits on our screen
			image_small = imutils.resize(blank_image, width=w)
			print("image_small", image_small.shape)
			
			# show the output image
			output_without_perceptron = output_without_perceptron + image_small 
		try:
			cv2.imwrite("predicted_plume.png", np.uint8(output_without_perceptron))
		except:
			print("Nothing detected")
		# cv2.waitKey(0)
