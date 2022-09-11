# setup
import sys
# Python 3.7 is required
assert sys.version_info >= (3,7)

import cv2 as cv
import numpy as np
import glob
import pandas as pd
import os
import argparse

# Make sure that optimization is enabled
if not cv.useOptimized():
    cv.setUseOptimized(True)

cv.useOptimized()

# argument parser
parser = argparse.ArgumentParser(description = "Traffic sign segmentation")
parser.add_argument('--input', help = "path to input image folder", default = 'Test_set', type = str)
args = parser.parse_args()


# read in the input folder
p = args.input + "/*.png"
path = glob.glob(p)

# read in the annotations for performance evaluation
data_dir = r"C:\Users\JianYong\Desktop\Test_set"
df = pd.read_csv(os.path.join(data_dir, 'annotations.csv'))

# to read the filename and annotations into a list
list_ground = [None]*100
n = 0

for index, row in df.sample(frac=1)[:].iterrows():
    inner = ['Test_set\\'+row.file_name, row.x1, row.y1, row.x2, row.y2]
    list_ground[n] = inner
    n = n + 1
        
# declaring list for storing the predicted bounding boxes
list_predicted = [None]*100
n = 0

# define kernel for the dilation and erosion operations
kernel_erode = np.ones((2, 2), np.uint8)
kernel_dilate = np.ones((3, 3), np.uint8)

# for trackbar
def empty(a):
    pass

cv.namedWindow("Parameters")
cv.resizeWindow("Parameters",640,240)
cv.createTrackbar("Threshold1","Parameters",23,255,empty)
cv.createTrackbar("Threshold2","Parameters",20,255,empty)

# calculate iou score
def get_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
    
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
    
    area_of_intersection = i_height * i_width
    
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
    
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
    
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    
    iou = area_of_intersection / area_of_union
    
    return iou

# main
for file in path:
    while 1:
        img = cv.imread(file)
        hh, ww = img.shape[:2]
        
        # this board is for showing the iou scored for each segmentation
        analysis_board = np.zeros((hh,ww), dtype='uint8')
        analysis_board = cv.cvtColor(analysis_board, cv.COLOR_GRAY2BGR)
        
        # blur the image
        imgBlur = cv.GaussianBlur(img, (7, 7), 1)
        
        # convert the image from bgr to gray
        imgGray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)
        
        # adjust the min and max thresholds for canny edge detection in real-time
        threshold1 = cv.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv.getTrackbarPos("Threshold2", "Parameters")
        imgCanny = cv.Canny(imgGray,threshold1,threshold2)
        
        # operations to join breaking parts and remove noise
        dilate = cv.dilate(imgCanny, kernel_dilate, iterations = 1)
        erode = cv.erode(dilate, kernel_erode, iterations = 1)

        # Identify contours
        contours, hierarchies = cv.findContours(erode,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        # Find the largest contour
        sorted_contours= sorted(contours, key=cv.contourArea, reverse= True)
        
        if(len(contours) >=2):
            # use the 2nd large contour as the largest contour 
            # as the findContours() will identify the whole image as one of the contour
            largest_item= sorted_contours[1]

            # when the largest contour (2nd) is too small, use back the 1st one
            if(cv.contourArea(largest_item) < ((img.shape[0]*img.shape[1])/11)):
                largest_item= sorted_contours[0]
                bounding_rect = cv.boundingRect(largest_item)
            bounding_rect = cv.boundingRect(largest_item)

            cv.rectangle(img, (int(bounding_rect[0]), int(bounding_rect[1])), \
                  (int(bounding_rect[0]+bounding_rect[2]), int(bounding_rect[1]+bounding_rect[3])), (0, 0, 255), 2)
        else:
            bounding_rect = (0, 0, 0, 0)
            
        # Draw white contour on black background as mask
        mask = np.zeros((hh,ww), dtype='uint8')
        cv.drawContours(mask, [largest_item], -1, (255,255,255), thickness=cv.FILLED)

        # fill the hole (flood fill operation)
        mask_u8 = np.uint8(mask)

        im_fill = mask_u8.copy()

        # define mask
        h, w = mask_u8.shape[:2]
        mask_f = np.zeros((h+2, w+2), np.uint8)

        # flood filling
        cv.floodFill(im_fill, mask_f, (0, 0), 255)

        # inverse the im_fill
        im_fill_inv = cv.bitwise_not(im_fill)

        # OR_operation
        ff_mask = im_fill_inv | mask_u8

        # mask the image
        img_mask=cv.bitwise_and(img, img, mask=ff_mask)

        # To compare
        imgCanny = cv.cvtColor(imgCanny, cv.COLOR_GRAY2BGR)
        erode = cv.cvtColor(erode, cv.COLOR_GRAY2BGR)
        dilate = cv.cvtColor(dilate, cv.COLOR_GRAY2BGR)
        ff_mask = cv.cvtColor(ff_mask, cv.COLOR_GRAY2BGR)
        imgGray = cv.cvtColor(imgGray, cv.COLOR_GRAY2BGR)
        
        # display the iou score for each segmentation result
        x=0
        for y in list_ground:
            if (list_ground[x][0][9:] == file[9:]):
                ground_truth_bbox = np.array([list_ground[x][1], list_ground[x][2], 
                                                list_ground[x][3], list_ground[x][4]], dtype=np.float32)
                prediction_bbox = np.array([int(bounding_rect[0]), int(bounding_rect[1]), 
                                            (int(bounding_rect[0])+int(bounding_rect[2])), 
                                            (int(bounding_rect[1])+int(bounding_rect[3]))], dtype=np.float32)
                iou = get_iou(ground_truth_bbox, prediction_bbox)
                iou=round(iou, 2)
                cv.putText(analysis_board, str(iou), (0 + 5, 0 + 15), cv.FONT_HERSHEY_COMPLEX, .6,
                                (0, 255, 0), 1)
            x=x+1
        
        # show the image
        cv.imshow(file[9:], np.hstack([img, imgGray, imgCanny, dilate, erode, ff_mask, img_mask, analysis_board]))
        
        # enter space bar to exit to the next image
        k = cv.waitKey(1) & 0xFF
        if k == 32:
            # save the predicted bounding box to the predicted list for model evaluation
            inner = [file, int(bounding_rect[0]), int(bounding_rect[1]), 
                 int(bounding_rect[0]+bounding_rect[2]), int(bounding_rect[1]+bounding_rect[3])]
            list_predicted[n] = inner
            n = n + 1
            break

cv.destroyAllWindows()    
    
n = 0
count = 0

for x in list_ground:
    m = 0
    for y in list_predicted:
        if (list_ground[n][0] == list_predicted[m][0]):
            ground_truth_bbox = np.array([list_ground[n][1], list_ground[n][2], 
                                          list_ground[n][3], list_ground[n][4]], dtype=np.float32)
            prediction_bbox = np.array([list_predicted[m][1], list_predicted[m][2], 
                                        list_predicted[m][3], list_predicted[m][4]], dtype=np.float32)
            iou = get_iou(ground_truth_bbox, prediction_bbox)
            print('Image ' + str(n+1) + ': ' + list_ground[n][0] + '\t\tIOU: ', round(iou, 2))
            if (iou >= 0.8):
                count = count + 1
            
        m = m + 1
    n = n + 1

print("\nTotal images used: " + str(n))
print("Total images that have been successfully segmented: " + str(count))
acc = round(count/(len(list_ground)), 2)
print("Model accuracy: " + str(acc))