# Image processing and loading utilities
import os
import numpy
import cv2

def myImageLoader(imageInput, enhance_for_office=False):
	# Convert PIL image to RGB first to ensure consistent format
	if hasattr(imageInput, 'convert'):
		imageInput = imageInput.convert('RGB')
	image = numpy.asarray(imageInput)
	
	# Office plan enhancement: binarize and thicken lines
	if enhance_for_office:
		# Convert to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		
	
		binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		
		# Remove noise
		kernel = numpy.ones((2,2), numpy.uint8)
		binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
		
		# Dilate to thicken lines - using a larger kernel for better connectivity
		kernel_dilate = numpy.ones((8,8), numpy.uint8)
		dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
		
		# Connect nearby lines that might be part of the same wall
		kernel_close = numpy.ones((7,7), numpy.uint8)
		dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
		
		# Convert back to black lines on white background
		dilated = cv2.bitwise_not(dilated)
		
		# Enhance contrast
		dilated = cv2.normalize(dilated, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		
		# Convert back to 3-channel RGB
		image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
	
	# Ensure the image is in the right data type and range
	if image.dtype != numpy.uint8:
		if image.max() <= 1.0:
			image = (image * 255).astype(numpy.uint8)
		else:
			image = image.astype(numpy.uint8)
	
	h, w, c = image.shape
	print(f"Processed image shape: {h}x{w}x{c}{' (office enhancement)' if enhance_for_office else ''}")
	return image, w, h

def getClassNames(classIds):
	result = list()
	for classid in classIds:
		data = {}
		if classid == 1:
			data['name'] = 'wall'
		if classid == 2:
			data['name'] = 'window'
		if classid == 3:
			data['name'] = 'door'
		result.append(data)	
	return result				


def normalizePoints(bbx, classNames):
	normalizingX = 1
	normalizingY = 1
	result = list()
	doorCount = 0
	index = -1
	doorDifference = 0
	for bb in bbx:
		index = index + 1
		if(classNames[index] == 3):
			doorCount = doorCount + 1
			if(abs(bb[3]-bb[1]) > abs(bb[2]-bb[0])):
				doorDifference = doorDifference + abs(bb[3]-bb[1])
			else:
				doorDifference = doorDifference + abs(bb[2]-bb[0])

		result.append([bb[0]*normalizingY, bb[1]*normalizingX, bb[2]*normalizingY, bb[3]*normalizingX])
	
	# Avoid division by zero
	if doorCount > 0:
		return result, (doorDifference/doorCount)
	else:
		return result, 0
		

def turnSubArraysToJson(objectsArr):
	result = list()
	for obj in objectsArr:
		data = {}
		data['x1'] = obj[1]
		data['y1'] = obj[0]
		data['x2'] = obj[3]
		data['y2'] = obj[2]
		result.append(data)
	return result


def calculateObjectArea(mask):
	"""Calculate the area of an object from its segmentation mask"""
	return numpy.sum(mask)

def calculateObjectCenter(bbox):
	"""Calculate the center point of a bounding box"""
	y1, x1, y2, x2 = bbox
	center_x = (x1 + x2) / 2
	center_y = (y1 + y2) / 2
	return {"x": float(center_x), "y": float(center_y)}

def encodeMaskSummary(mask):
	"""Create a summary of the segmentation mask instead of full encoding"""
	# Count non-zero pixels and get bounding box of the mask
	non_zero_pixels = numpy.sum(mask > 0)
	total_pixels = mask.shape[0] * mask.shape[1]
	coverage_percentage = (non_zero_pixels / total_pixels) * 100
	
	# Find bounding box of non-zero region
	rows = numpy.any(mask, axis=1)
	cols = numpy.any(mask, axis=0)
	
	if numpy.any(rows) and numpy.any(cols):
		rmin, rmax = numpy.where(rows)[0][[0, -1]]
		cmin, cmax = numpy.where(cols)[0][[0, -1]]
		mask_bbox = {"x1": int(cmin), "y1": int(rmin), "x2": int(cmax), "y2": int(rmax)}
	else:
		mask_bbox = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
	
	return {
		"coverage_percentage": float(coverage_percentage),
		"non_zero_pixels": int(non_zero_pixels),
		"total_pixels": int(total_pixels),
		"mask_bbox": mask_bbox
	}

def getClassName(classId):
	"""Get class name from class ID"""
	class_map = {1: 'wall', 2: 'window', 3: 'door'}
	return class_map.get(classId, 'unknown')