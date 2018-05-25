import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog

# CORE FUNCTIONS FOR VEHICLE DETECTION PROVIDED IN LESSONS
class hog_channels:
    CH0 = 1 << 0
    CH1 = 1 << 1
    CH2 = 1 << 2

# Function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Function that estimates bin range depending on the color and datatype
# please check https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html for information
def color2bin_range(color_space='RGB'):
    # First, check datatype
    ch0_range = ch1_range = ch2_range = (0, 256)
    if (color_space == 'HSV') | (color_space == 'HLS'):
        ch0_range = (0, 180)
            
    return ch0_range, ch1_range, ch2_range

# Function to compute color histogram features 
def color_hist(img, nbins=32, ch0_range=(0, 256), ch1_range=(0, 256), ch2_range=(0, 256)):
    
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=ch0_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=ch1_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=ch2_range)
    
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features, channel1_hist, channel2_hist, channel3_hist

# Function to extract features from an image
def extract_features(image, color_space='RGB', 
                     spatial_size=(32, 32), hist_bins=32, 
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    file_features = []

    if color_space != 'RGB':
        feature_image = cv2.cvtColor(image, eval('cv2.COLOR_RGB2'+color_space))
    else: 
        feature_image = np.copy(image)      
    # Spatial features
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    
    # Histogram features
    if hist_feat == True:
        # Identify correct histogram bin ranges for each channel
        ch0_hist_range, ch1_hist_range, ch2_hist_range = color2bin_range(color_space=color_space)
        hist_features = color_hist(feature_image, nbins=hist_bins, 
                                   ch0_range=ch0_hist_range, ch1_range=ch1_hist_range, ch2_range=ch2_hist_range)[0]
        file_features.append(hist_features)
    
    # HOG features
    if hog_feat == True:
        hog_features = []
        if (hog_channel & hog_channels.CH0) != 0:
            hog_features.extend(get_hog_features(feature_image[:,:,0], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
        if (hog_channel & hog_channels.CH1) != 0:
            hog_features.extend(get_hog_features(feature_image[:,:,1], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
        if (hog_channel & hog_channels.CH2) != 0:
            hog_features.extend(get_hog_features(feature_image[:,:,2], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
        
        file_features.append(hog_features)

    return file_features
    
# Function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img