#!/usr/bin/env python

import cv2

img_file_prefix = "/home/adam/Desktop/source_images_rotated_cropped/source_"
export_prefix = "/home/adam/Desktop/source_images_rotated_cropped/tiles/"

index = 1
final_index = 35

while index <= final_index:
    img_file = img_file_prefix + str(index) + '_rotated_cropped.jpg'
    print 'next source image +++++++++++++++++++++++'
    print img_file
    
    img = cv2.imread(img_file)

    height, width, channels = img.shape

    num_cols = height // 512
    num_rows = width // 512

    print 'tiling....................................'
    for x in range(0, num_rows):
        for n in range(0, num_cols):
	    name = img_file_prefix + str(index) + '_' + str(n) + '_' + str(x) + '.jpg'
	    col_pixel_start = (512 * n) + 1
	    row_pixel_start = (512 * x) + 1
	    tile = img[col_pixel_start:(col_pixel_start + 512), row_pixel_start:(row_pixel_start + 512)]
            print name

	    split_name = name.split('/')
	    output_name = split_name[5]
	    cv2.imwrite(export_prefix + output_name, tile)

    index = index + 1

