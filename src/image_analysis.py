import cv2 as cv
import numpy as np
from skimage.exposure import rescale_intensity
# from subprocess import call
# call(["python setup.py build_ext --inplace"], shell=True)
from convolve import convolution as conv2
import time

path = r"C:\Users\abdul\Desktop\CS 512 - Computer vision\Homework\HW_1\AS1\data\home.jpg"

def read_image(path):
    img = cv.imread(path)
    print("image shape: ", img.shape)
    return img

def show_image(window_name,img,wait_key=0):
    cv.imshow(window_name,img)

def bgr_to_gray(img):
    out = (0.07 * img[:,:,2] + 0.72 * img[:,:,1] + 0.21 * img[:,:,0]).astype(np.uint8)
    return out

def smoothing_slider_1(win_name='gray'):
    pass

def rotate_slider():
    pass



def rotation(img, theta):
    height, width = img.shape[0], img.shape[1]
    img_center = (width/2, height/2)
    rot_matrix = cv.getRotationMatrix2D(img_center, theta, 1.)

    # absolute value of the sin and cos of the rotation angle
    cosine = abs(rot_matrix[0,0])
    sine = abs(rot_matrix[0,1])

    # New height and width after rotation
    new_w = int(height * sine + width * cosine)
    new_h = int(height * cosine + width * sine)

    # reorienting the new center coordinates
    rot_matrix[0, 2] += -img_center[0] + new_w/2
    rot_matrix[1, 2] += -img_center[1] + new_h/2

    img_rot = cv.warpAffine(img, rot_matrix, (new_w, new_h))
    return img_rot


def convolution(img, kernel):
    # kernel height and width
    kernel_h, kernel_w = kernel.shape[0], kernel.shape[1]
    # image height and width
    img_h, img_w = img.shape[0], img.shape[1]
    # initialize the result of the convolution
    result = np.zeros((img_h, img_w), np.float32)
    # padding with replication
    padding = (kernel_h - 1) // 2
    img = cv.copyMakeBorder(img, padding,padding,padding,padding, cv.BORDER_REPLICATE)
    # convolution operation
    for h in range(padding, img_h+padding):
        for w in range(padding, img_w+padding):
            section = img[h-padding:h+padding+1, w-padding:w+padding+1]
            conv_value = (section * kernel).sum()
            result[h-padding, w-padding] = conv_value

    # rescale to required pixel range
    result = rescale_intensity(result, in_range=(0,255))
    result = (result*255).astype('uint8')

    return result


def run(path):
    img = read_image(path)
    show_image('img', img)
    key = cv.waitKey(0) & 0xFF
    color_channel_counter = 0

    while (True):
        if key == ord('i'):
            print('<ESC>: quit')
            a = cv.waitKey(0)
            if a == 27:
                cv.destroyAllWindows()
                img = read_image(path)
                show_image('img', img)
                key = cv.waitKey(0)
                continue

        if key == ord('w'):
            cv.imwrite('out.jpg', img)
            cv.destroyAllWindows()
            img = read_image(path)
            show_image('img', img)
            key = cv.waitKey(0)
            continue

        if key == ord('g'):
            img = read_image(path)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            cv.destroyAllWindows()
            show_image('img_gray',img_gray)
            key = cv.waitKey(0)
            img = img_gray
            continue

        if key == ord('a'):
            img = read_image(path)
            img_gray = bgr_to_gray(img)
            cv.destroyAllWindows()
            show_image('img_grey', img_gray)
            key = cv.waitKey(0)
            img = img_gray
            continue

        if key == ord('c'):
            img = read_image(path)
            color_channel_counter = color_channel_counter % 3
            b = img.copy()
            if color_channel_counter == 0:
                b[:,:,1] = 0
                b[:,:,2] = 0
            elif color_channel_counter ==1:
                b[:, :, 0] = 0
                b[:, :, 2] = 0
            else:
                b[:, :, 0] = 0
                b[:, :, 1] = 0

            color_channel_counter += 1
            show_image('img_one_channel', b)

            key = cv.waitKey(0)
            img = b
            continue

        if key == ord('s'):
            img = read_image(path)
            cv.destroyAllWindows()
            print('\nslide the trackbar to smooth')
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            print('press <ESC> to exit smoothing mode')
            cv.namedWindow('gray')
            cv.createTrackbar('bar', 'gray', 0, 10, smoothing_slider_1)

            while True:
                n = cv.getTrackbarPos('bar', 'gray')
                img_blur = cv.blur(img_gray, (n+1, n+1))
                show_image('gray', img_blur)
                key = cv.waitKey(1)
                if key == 27:
                    break

            key = cv.waitKey(0)
            img = img_blur
            continue

        if key == ord('b'):
            img = read_image(path)
            cv.destroyAllWindows()
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            print('\npress <enter> after sliding the bar to commit')
            print('press <ESC> to exit smoothing mode')
            cv.namedWindow('grey')
            cv.createTrackbar('bar', 'grey', 0, 4, smoothing_slider_1)

            while True:
                n = cv.getTrackbarPos('bar', 'grey')
                if n == 0:
                    kernel = np.ones((n + 1, n + 1), np.float32) * (1.0 / ((n + 1) * (n + 1)))
                elif n == 1:
                    kernel = np.ones((3,3), np.float32) * (1.0 / ((3) * (3)))
                elif n == 2:
                    kernel = np.ones((5,5), np.float32) * (1.0 / ((5) * (5)))
                elif n == 3:
                    kernel = np.ones((7,7), np.float32) * (1.0 / ((7) * (7)))
                else: kernel = np.ones((9,9), np.float32) * (1.0 / ((9) * (9)))

                t1 = time.time()
                img_blur = conv2(img_gray, kernel)
                t2 = time.time()
                print ('time taken ', t2-t1)
                show_image('grey', img_blur)
                key = cv.waitKey(0)
                if key == 27:
                    break

            key = cv.waitKey(0)
            img = img_blur
            continue

        if key == ord('d'):
            img = read_image(path)
            ratio = 0.5
            img = cv.resize(img, (0,0), fx=ratio, fy=ratio, interpolation=cv.INTER_NEAREST)
            cv.destroyAllWindows()
            show_image('downsample_no_smooth', img)
            key = cv.waitKey(0)
            continue

        if key == ord('e'):
            img = read_image(path)
            img = cv.blur(img, (5,5))
            ratio = 0.5
            img = cv.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv.INTER_NEAREST)
            cv.destroyAllWindows()
            show_image('downsample_with_smooth', img)
            key = cv.waitKey(0)
            continue

        if key == ord('x'):
            img = read_image(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            x_filter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) * 1/8
            img = conv2(img, x_filter)
            img = rescale_intensity(img, in_range=(0, 255))
            img = (img * 255).astype('uint8')
            print(img)
            cv.destroyAllWindows()
            show_image('x_derivative', img)
            key = cv.waitKey(0)
            continue

        if key == ord('y'):
            img = read_image(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            y_filter = np.array([[1,2,1], [0,0,0], [-1, -2,-1]]) * 1/8
            img = conv2(img, y_filter)
            img = rescale_intensity(img, in_range=(0, 255))
            img = (img * 255).astype('uint8')
            print(img)
            show_image('y_derivative', img)
            key = cv.waitKey(0)
            continue

        if key == ord('m'):
            img = read_image(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            sobel_x = cv.Sobel(img, cv.CV_64F, 1,0,ksize=3)
            sobel_y = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
            img = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
            img = rescale_intensity(img, in_range=(0, 255))
            img = (img * 255).astype('uint8')
            print(img)
            show_image('image_gradient',img)
            key = cv.waitKey(0)
            continue

        if key == ord('r'):
            img = read_image(path)
            cv.destroyAllWindows()
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            print('\npress <enter> after sliding bar to commit')
            print('press <ESC> to exit rotation mode')
            cv.namedWindow('rotate')
            cv.createTrackbar('rotate_bar', 'rotate', 0, 7, smoothing_slider_1)

            while True:
                angle = cv.getTrackbarPos('rotate_bar', 'rotate')
                new = rotation(img, 45*angle)
                show_image('rotate', new)
                key = cv.waitKey(0)
                if key == 27:
                    print('rotation mode closed. Press i to go to main menu or w to save image')
                    break

            key = cv.waitKey(0)
            img = new
            continue

        if key == ord('h'):
            print("\nThis program involves image manipulations with openCv. Use the keys as described"
                  "below to perform various operations on the image.\n"
                  "NOTE: Always look at the console for possible additional instructions when running "
                  "the program.\n'h': press h to view these instructions\n"
                  "'i': clears all edits to the image. (It'll prompt you to press escape to commit)\n"
                  "'w': saves the current state of the image as 'out.jpg'\n"
                  "'g': converts an image to grayscale\n"
                  "'a': converts an image to grayscale using my implementation\n"
                  "'c': cycles through the different color channels each time it's pressed\n"
                  "'s': converts an image to grayscale and applies smoothing (with a trackbar)\n"
                  "'b': converts an image to grayscale and applies smoothing (with a trackbar) using"
                  "my implemented function\n"
                  "'d': downsample by a factor of 2\n"
                  "'e': downsample by a factor of 2 then apply smoothing\n"
                  "'x': converts the image to grayscale and convolves it with an x-derivative filter\n"
                  "'y': converts the image to grayscale and convolves it with an y-derivative filter\n"
                  "'m': displays the image with the gradient magnitude (x- and y-)\n"
                  "'r': converts the image to grayscale and uses a trackbar to control amount of rotation\n")

            key = cv.waitKey(0)
            continue



        else: break


if __name__ == '__main__':
    run(path)


