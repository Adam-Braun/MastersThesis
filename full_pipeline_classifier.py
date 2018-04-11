#!/usr/bin/python3

import numpy
import time
import cv2
import mvnc.mvncapi as mvnc

graph_file = 'graph2'  # file name of compiled neural network
source_image = 'source_17_rotated_cropped.jpg'  # name of image file to be processed

# average pixel color of training dataset
average_pixel = numpy.float16([135.42515564, 143.031448364, 141.488006592])

# image overlays to be applied in post processing to visualize data output
blue_overlay = numpy.zeros((512, 512, 3), numpy.uint8)
blue_overlay[:] = (255, 0, 0)

red_overlay = numpy.zeros((512, 512, 3), numpy.uint8)
red_overlay[:] = (0, 0, 255)

green_overlay = numpy.zeros((512, 512, 3), numpy.uint8)
green_overlay[:] = (0, 255, 0)

yellow_overlay = numpy.zeros((512, 512, 3), numpy.uint8)
yellow_overlay[:] = (0, 255, 255)

# empty list to hold how long each inference takes
inference_time_list = []


def open_ncs_device():

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print("No devices found")
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device(devices[0])
    device.OpenDevice()

    return device


def load_graph(device):

    # Read the graph file into a buffer
    with open(graph_file, mode='rb') as f:
        g_file = f.read()

    # Load the graph buffer into the NCS
    graph = device.AllocateGraph(g_file)

    return graph


def pre_process_image(img):
    # process the image to 224x224 pixels because that is was the input layer of GoogleNet requires
    img = cv2.resize(img, (int(224), int(224)))

    # convert image to fp16 data type and perform mean pixel subtraction on each pixel
    img = img.astype(numpy.float16)
    img = (img - numpy.float16(average_pixel))

    # return pre-processed image
    return img


def infer_image(graph, img, image):

    # Labels used for classification output
    labels = ['city', 'clouds', 'other', 'water']

    # Load the image as a half-precision floating point array
    graph.LoadTensor(img, 'user object')

    # Get the results from NCS
    output, userobj = graph.GetResult()

    # Get execution time
    inference_time = numpy.sum(graph.GetGraphOption(mvnc.GraphOption.TIME_TAKEN))
    inference_time_list.append(inference_time)

    # Get classification inference and print to screen
    top_prediction = output.argmax()
    print(labels[top_prediction])

    # Read tile that has been saved to file for post processing
    image = cv2.imread('staging.jpg')

    # Tint tile to color representing class it belongs to
    if labels[top_prediction] == 'city':
        image = cv2.addWeighted(image, 0.8, red_overlay, 0.15, 0)

    elif labels[top_prediction] == 'water':
        image = cv2.addWeighted(image, 0.8, blue_overlay, 0.15, 0)

    elif labels[top_prediction] == 'clouds':
        image = cv2.addWeighted(image, 0.8, yellow_overlay, 0.15, 0)

    else:
        image = cv2.addWeighted(image, 0.8, green_overlay, 0.15, 0)

    # return post processed image
    return image


def close_ncs_device(device, graph):
    # use NCSDK to close Neural Compute Stick
    graph.DeallocateGraph()
    device.CloseDevice()


def main():
    start_time = time.time()  # note start time of script for timing

    # open Movidius Neural Compute Stick
    device = open_ncs_device()

    # Load compiled GoogleNet classification network onto device
    graph = load_graph(device)

    # read image into script
    image_file = cv2.imread(source_image)

    # determine the size of the image loaded and calculate how many tiles will be produced from image
    height, width, channels = image_file.shape
    num_cols = height // 512  # tiles will be 512 pixels wide
    num_rows = width // 512  # tiles will be 512 pixels tall

    number_of_inferences = 0  # index for how many tiles are produced

    # tile the large input image into 512x512 pixel images
    for x in range(num_rows):
        for n in range(num_cols):
            col_pixel_start = (512 * n) + 1
            row_pixel_start = (512 * x) + 1
            tile = image_file[col_pixel_start:(col_pixel_start + 512), row_pixel_start:(row_pixel_start + 512)]

            cv2.imwrite('staging.jpg', tile)  # write tile to file to be used in post processing

            img = pre_process_image(tile)  # process image to be used in neural network
            output_image = infer_image(graph, img, image_file)  # add post processed tile to row of final image
            if n == 0:
                row_img = output_image

            else:
                row_img = numpy.concatenate((row_img, output_image), axis=0)

            number_of_inferences = number_of_inferences + 1

        # add the rows of post processed imaage into final processed image
        if x == 0:
            final_img = row_img

        else:
            final_img = numpy.concatenate((final_img, row_img), axis=1)

    close_ncs_device(device, graph)  # close the Movidius Neural Compute Stick
    cv2.imwrite('output_image.jpg', final_img)  # save the output image to a file

    # print out information about processing time, etc.
    print ('Completed in: ')
    print(time.time() - start_time)

    print('Total number of tiles classified')
    print(number_of_inferences)
    print('Total milliseconds spent in inference: ')
    print(sum(inference_time_list))
    print('Average inference time (in milliseconds): ')
    print(numpy.mean(inference_time_list))


if __name__ == '__main__':
    main()
