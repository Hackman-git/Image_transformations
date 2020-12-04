# Image_transformation
Images need to be transformed and analyzed for a variety of reasons which include feature detection, graphics processing, and so on. This enables us to extract meaningful information from images which can be capitalized on in various applications. In this report, we touch on some basic image processing techniques regarding edge detection, using the OpenCv computer vision library for python, and we also evaluate the correctness of the algorithms implemented.

The algorithms/operations implemented (without openCv functions) include:
1. Converting an image to grayscale
2. Cycling through color channels of an image
3. Performing convolution
4. Applying smoothing to a grayscale image by convolving with a smoothing filter
5. Applying an x-derivative filter to a grayscale image
6. Applying a y-derivative filter to a grayscale image
7. Combining x-derivative and y-derivative filters to obtain image gradients
8. Rotating an image by an angle and avoiding crop
