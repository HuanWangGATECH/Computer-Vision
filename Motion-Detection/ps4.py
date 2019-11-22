"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os

# from scipy.signal import convolve2d


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3,scale=0.125)
    return sobelx

    raise NotImplementedError


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3,scale=0.125)
    return sobely

    raise NotImplementedError


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    filt = (k_size, k_size)
    if k_type == 'gaussian':
        temp_a = cv2.GaussianBlur(img_a, ksize=filt, sigmaX=sigma, sigmaY=sigma)
        temp_b = cv2.GaussianBlur(img_b, ksize=filt, sigmaX=sigma, sigmaY=sigma)
    else:
        temp_a = np.copy(img_a)
        temp_b = np.copy(img_b)
        
    It = cv2.subtract(temp_a, temp_b).astype(np.float64)
    Ix = gradient_x(temp_a)
    Iy = gradient_y(temp_a)

    Sxx = cv2.boxFilter(Ix**2, -1, ksize=filt, normalize=False)
    Syy = cv2.boxFilter(Iy**2, -1, ksize=filt, normalize=False)
    Sxy = cv2.boxFilter(Ix*Iy, -1, ksize=filt, normalize=False)
    Sxt = cv2.boxFilter(Ix*It, -1, ksize=filt, normalize=False)
    Syt = cv2.boxFilter(Iy*It, -1, ksize=filt, normalize=False)

    M_det = np.clip(Sxx * Syy - Sxy ** 2, 0.000001, np.inf)
    temp_u = -1 * (Syy * (-Sxt) + (-Sxy) * (-Syt))
    temp_v = -1 * ((-Sxy) * (-Sxt) + Sxx * (-Syt))
    U = np.where(M_det != 0, temp_u / M_det, 0).astype(np.float64)
    V = np.where(M_det != 0, temp_v / M_det, 0).astype(np.float64)

    return (U,V)

    raise NotImplementedError

def generating_kernel(a):
  w_1d = np.array([0.25 - a/2.0, 0.25, a, 0.25, 0.25 - a/2.0])
  return np.outer(w_1d, w_1d)


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """

    ans = None
    kernel = generating_kernel(0.4)
#    temp_out = scipy.signal.convolve2d(image,kernel,'same')
    temp_out = cv2.filter2D(image, -1, kernel)
    out = temp_out[::2,::2]
    return out

    raise NotImplementedError


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """

    ans = [image]
    temp = np.copy(image)
    for i in range(1,levels):
        temp = reduce_image(temp)
        ans.append(temp)
    return ans

    raise NotImplementedError


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape
    hb,wb = imgb.shape
    new_img = np.zeros(shape=(np.max([ha, hb]), wa+wb))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """



    out_img = normalize_and_scale(img_list[0])
    i = 0
    while i+1 <len(img_list): 
        out_img = concat_images(out_img,normalize_and_scale(img_list[i+1]))
        i += 1
    # cv2.imshow('test',out_img)
    # cv2.waitKey(0)
    return out_img


    raise NotImplementedError


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """

    ans = None
    kernel = np.array([[0.125,0.5,0.75,0.5,0.125]], dtype=np.float64)#generating_kernel(0.4)
    kernel = np.outer(kernel, kernel)
    temp_out = np.zeros((len(image)*2, len(image[0])*2), dtype=np.float64)
    temp_out[::2,::2]=image[:,:]
    ans = cv2.filter2D(temp_out, -1, kernel, borderType=cv2.BORDER_REFLECT101) 
    # ans = 4*convolve2d(temp_out,kernel,'same')

    return ans

    raise NotImplementedError


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    ans = []
    for i in range(len(g_pyr)-1):
        g_pre = g_pyr[i]
        l_exp = expand_image(g_pyr[i+1])
        if g_pre.shape[0] < l_exp.shape[0]:
            l_exp = np.delete(l_exp, (-1), axis=0)
        if g_pre.shape[1] < l_exp.shape[1]:
            l_exp = np.delete(l_exp, (-1), axis=1)
        ans.append(g_pre - l_exp)
    ans.append(g_pyr[-1])
    return ans

    raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    ny, nx = image.shape
    X, Y = np.meshgrid(range(nx), range(ny))
    map_X = (X + U).astype(np.float32)
    map_Y = (Y + V).astype(np.float32)
    warped = cv2.remap(image, map_X, map_Y, interpolation=interpolation, borderMode=border_mode)
    return warped

    raise NotImplementedError


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    img_a_gaussian = gaussian_pyramid(img_a,levels)
    img_b_gaussian = gaussian_pyramid(img_b,levels)

    h,w = img_a.shape
    h = h // (2**(levels-1))
    w = w // (2**(levels-1))

    U,V = np.zeros((h,w), dtype = np.float64),np.zeros((h,w), dtype = np.float64)

    for i in range(levels-1, -1, -1):
        if i != levels-1:
            U = 2 * expand_image(U)
            V = 2 * expand_image(V)
        img_B = img_b_gaussian[i]

        if i == levels-1:
            img_B_wrap = img_B
        else:
            img_B_wrap = warp(img_B, U, V, interpolation, border_mode)

        img_A = img_a_gaussian[i]

        u,v = optic_flow_lk(img_A, img_B_wrap, k_size, k_type, sigma)
        U += u
        V += v
    
    return (U,V)

    raise NotImplementedError

def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None
    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    raise NotImplementedError
