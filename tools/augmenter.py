#!/usr/bin/env python

from PIL import Image
import glob

target_directory = '/home/adam/thesis/images/tiles_batch1/water'

for filename in glob.glob(target_directory + '/*.jpg'):

	split_name = filename.split('/')
        name = split_name[7].split('.')
        name = name[0]
        print name

	img = Image.open(filename)
	rotated_image = img.rotate(90)
	rotated_image.save(name + '_90.jpg')

	rotated_image = img.rotate(180)
        rotated_image.save(name + '_180.jpg')

	rotated_image = img.rotate(270)
        rotated_image.save(name + '_270.jpg')

	flipped_image = img.transpose(Image.FLIP_LEFT_RIGHT)
	flipped_image.save(name + '_flippedLR.jpg')

        flipped_image = img.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_image.save(name + '_flippedTB.jpg')
