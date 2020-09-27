from PIL import Image
import cv2 as cv
import numpy as np
import glob
import os

desired_size = (256,256)
current_dir = os.getcwd()

input_folder_name = current_dir + "/sample-data/"
output_folder_name = current_dir + "/processed-data/"

# the regex that defines what images to process
files_regex ="*.png"

#Creating a new directory if does not exist
def createFolder():
    '''
    created the new folder to hold the new processed images.
    if the folder already exists does nothing and the data will be added to the currently existing folder
    '''
    try:
        if not os.path.isdir(output_folder_name):
            os.makedirs(output_folder_name, 777, exist_ok=True)
            print ("Successfully created the directory %s " % output_folder_name)
        else:
            print ("folder %s already exists, writing to the existing file. " % output_folder_name)
    except OSError:
        print ("Creation of the directory %s failed" % output_folder_name)
        

def processData():
    '''
    processes each images from input folder and saves it to output folder 
    '''
    test_filelist = glob.glob(input_folder_name + files_regex) # returns a list of all the files in the folder that match the regex

    for i in range(len(test_filelist)):
        im = cv.imread(test_filelist[i])

        filename = test_filelist[i].split('\\')[-1]

        #resize images to the desired size
        new_im = cv.resize(im, desired_size)

        # turn to gray scale
        new_im = cv.cvtColor(new_im, cv.COLOR_BGR2GRAY)

        # CLAHE transformation
        clahe = cv.createCLAHE(clipLimit = 3) 
        new_im = clahe.apply(new_im)

        Image.fromarray(new_im).save(os.path.join(output_folder_name, 'grayscale-processed-' + filename), "PNG")


def main():
    createFolder()
    processData()
    

if __name__ == "__main__":
    main()