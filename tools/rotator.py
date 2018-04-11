#!/usr/bin/env python

from PIL import Image
import glob

target_directory = '/home/adam/Desktop/source_images'
# filename = '/home/adam/Desktop/source_images/source_14.tif'

for filename in glob.glob(target_directory + '/*.tif'):
	split_name = filename.split('/')
	name = split_name[5].split('.')
	name = name[0]
	print name

	img = Image.open(filename)
	rotated_image = img.rotate(348)
	# rotated_image.show()
	rotated_image.save(name + '_rotated.jpg')
