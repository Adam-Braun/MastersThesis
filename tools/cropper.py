#!/usr/bin/env python

from PIL import Image
import glob

target_directory = '/home/adam/Desktop/source_images_rotated'
# filename = '/home/adam/Desktop/source_images/source_14.tif'

for filename in glob.glob(target_directory + '/*.jpg'):
	split_name = filename.split('/')
	name = split_name[5].split('.')
	name = name[0]
	print name

	img = Image.open(filename)
	cropped_image = img.crop((200, 900, 8700, 3600))
	# rotated_image.show()
	cropped_image.save(name + '_cropped.jpg')
