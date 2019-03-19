## **Finding Lane Lines on the Road** 

#### By Katherine Lohff

---

**Objective**

While driving around, I pay attention to a variety of demarcations on the road to safely navigate my car. In this first project, I used a simple method to detect lane lines in the provided test images/videos using Python and OpenCV (an Open Source Computer Vision Library). In the following sections, I will discuss the steps I implemented and offer improvements.


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1] TODO: fix images!

---

### Environment Set-Up

Installed Docker (program that manages container, provides a self-contained reproducable environment)

```
$ docker pull udacity/carnd-term1-starter-kit

$ docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit


```

Edited Jupyter Notebook in my browser (at address 127.0.0.1).

### Reflection

#### Overview

I created a pipeline and tested it first on individual images, then on a video stream (which is really just a series of images).



Import packages and read in image/frame of video

My pipeline consisted of ? general steps:
2. Convert to grayscale
3. Apply Gaussian Blur function
4. Run Canny Edge Detection
5. Apply a Polygon Mask
6. Run Hough Transform on masked image/frame
7. Extrapolate between dotted line segments
Smooth Between Frames (so they aren't jittery, and because in some frames it can't detect a line)
8. Draw solid lines over lane lines

Save output

TODO: In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...


#### Step-By-Step

**Import Useful Packages**

```python

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    import cv2
    %matplotlib inline

```

**Read In Image**

```python

    image = mpimg.imread('test_images/solidWhiteRight.jpg')
    
    #print out some stats and plot image
    print('This image is:', type(image), 'with dimensions:', image.shape)
    plt.imshow(image)

```
![solidWhiteRight.jpg][image1]

This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3)

**Convert to Grayscale**

```python

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
```
![Grayscale Version][image2]

**Apply Gaussian Blur**

```python

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
```

![Gaussian Blur][image3]

**Run Canny Edge Detection**

```python

    # Define parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

```

![Canny Edge][image4]

**Apply a Polygon Mask**

This function applies an image mask to define the region of interest as the area inside the polygon formed from `vertices`. The rest of the image is set to black.`vertices` is a numpy array of integer points where the points are defined by...(TODO)

```python

    mask = np.zeros_like(edges)   
    ignore_mask_color = 255
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(480, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

```

![Polygon Mask][image5]

**Run Hough Transform**

Run the Hough Transform on the output of Masked Edges. (TODO: Explain Hough)

```python

    line_image = np.copy(image)*0 # creating a blank
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 # minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

```

![Hough Transform][image6]

**Extrapolate between dotted line segments**

If the lane line was solid, it was straightforward to draw a line on top of it. However, if the lane line happened to be segmented, an additional step was necessary to extrapolate a solid line on top of that segmented lane line.

```python

```

![Extrapolate][image7]

**Draw solid lines over lane lines and save output**

Return the final output (image where lines are drawn on lanes) and save to (TODO: location)


```python

```

![Final Image][image8]

### 2. Identify potential shortcomings with your current pipeline and 3. Suggest possible improvements to your pipeline

My code takes an average of two slopes and intercepts (the slope/intercept of the current frame and the slope/intercept of the previous frame). I could significantly improve this if I kept around the average slope/intercept from many (at least 20+) previous frames. and used this with the slope/intercept of the current frame (weighted 95% old, 5% current)

There is a line (TODO: insert line) in the code which tests for a line and the way it is written it would breaks on first frame of video if no lines are detected in that frame. (Fix: Could add a conditional statement to handle the case if slope_right_list or intercept_right_list are empty.)

I decided to average lane slopes and intercepts across frames to smooth the draw_lines function, a major improvement would be to discard any line that deviates too much from the computed average from previous frames.

I used (poly_fit) function of degree 1 (form y = mx + b) which describes a straight line. This works fairly well for the first two videos but in the challenge video the lane lines have considerably more curvature. A future imporovement would be to use a polynomial fit of (greater than 1) degree to improve my fit to these curvy lane lines. 





