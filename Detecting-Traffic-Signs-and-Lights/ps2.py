"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np

class Line:
    def __init__(self, line):
        self.line = line
        self.length = np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2)
        denominator = line[2] - line[0]
        if denominator == 0:
            self.angle = 90
        else:
            self.angle = np.arctan((line[3] - line[1])/denominator)/np.pi * 180
        self.mid = ((line[2] + line[0])/2, (line[3] + line[1])/2)

def get_tl(circles):
	n = len(circles[0,:])
	thresh = 10
	result = []
	for i in range(n):
		for j in range(i+1,n):
			for k in range(j+1,n):
				if abs(circles[0,i][0] - circles[0,j][0]) < thresh and abs(circles[0,j][0] - circles[0,k][0]) < thresh:
					if abs(circles[0,i][2] - circles[0,j][2]) < thresh and abs(circles[0,j][2] - circles[0,k][2]) < thresh:

						result.append(circles[0,i])
						result.append(circles[0,j])
						result.append(circles[0,k])
	# print(result)
						result = sorted(result, key = lambda y: y[1])
	# print(result)
						return result
	return



def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """

    img_temp = np.copy(img_in)
    image_cols, image_rows, _ = img_temp.shape
    # img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img_temp_gray', img_temp)
    img_canny = cv2.Canny(img_temp, 110, 60)
    # cv2.imshow('img_temp_gray', img_temp)
    circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT,1, 20, param1 = 50,param2 = 30,minRadius = 5,maxRadius = 50)
    if circles is None: return ((0,0),'null')
    circle_tl = get_tl(circles)
    # print(circle_tl)
    # circle_tl = circles[0,:]
    if circle_tl is None: return ((0,0),'null')
    for circle in circle_tl:
    	col = int(circle[0])
    	row = int(circle[1])
    	# cv2.circle(img_temp, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
    	if img_temp[row, col, :][2] == img_temp[row, col, :][1] and img_temp[row, col, :][2] > 250: color = 'yellow'
    	elif img_temp[row, col, :][2] > 250 and img_temp[row, col, :][1] < 200: color = 'red'
    	elif img_temp[row, col, :][1] > 250 and img_temp[row, col, :][2] < 200: color = 'green'
    # cv2.imshow('img_temp_gray', img_temp)
    return ((circle_tl[1][0],circle_tl[1][1]), color)

    
    # 	print(img_temp[int(circle[1]), int(circle[0]), :])

    # hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSL)
    # lower_yellow = np.array([20,0,0])
    # upper_yellow = np.array([40,255,255])
    # mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # res = cv2.bitwise_and(img_temp,img_temp, mask= mask)
    # cv2.imshow('img_temp', hsv)
    # for circle in circles[0,:]:
    #  	print(hsv[int(circle[1]), int(circle[0]), :])
    raise NotImplementedError


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    img_temp = np.copy(img_in)
    img_red = img_temp[:,:,2]
    img_gre = img_temp[:,:,1]
    img_blu = img_temp[:,:,0]
    # img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    img_bi = np.zeros_like(img_red)
    img_bi[(img_red >= 200) & (img_red <= 255)& (img_gre >= 200) & (img_gre <= 255) & (img_blu >= 200)] = 255
    # img_bi[(img_gre >= 200) & (img_gre <= 255)] = 0
    img_canny = cv2.Canny(img_bi, 110, 60)
    # cv2.imshow('img_temp', img_bi)
    lines = cv2.HoughLinesP(img_canny, rho = 1, theta = np.pi/180*30, threshold = 50, minLineLength = 50, maxLineGap = 4)
    # print(lines)
    if lines is None: return (0,0)
    N = lines.shape[0]
    xmid,ymid = [],[]
    # print(N)

    for line in lines:
        line_temp = line.flatten()
        line_para = Line(line_temp)

        if (line_para.angle > 40 and line_para.angle < 80) or (line_para.angle < -40 and line_para.angle > -80) :
            x1 = line[0][0]
            x2 = line[0][2]
            y1 = line[0][1]
            y2 = line[0][3]
            xmid.append((x1+x2) / 2)
            ymid.append((y1+y2) / 2)
    #         cv2.line(img_temp,(x1,y1),(x2,y2),(255,0,0))
    # cv2.imshow('img_temp',img_temp)

    if len(xmid) == 0: return (0,0)
    center = (int(np.mean(xmid)), int(np.mean(ymid) - 5))
    return center
    raise NotImplementedError

def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    img_temp = np.copy(img_in)
    img_red = img_temp[:,:,2]
    img_gre = img_temp[:,:,1]
    img_blu = img_temp[:,:,0]
    img_bi = np.zeros_like(img_red)
    img_bi[(img_red >= 200) & (img_gre < 100) & (img_blu < 100)] = 255
    img_canny = cv2.Canny(img_bi, 10, 0)
    # cv2.imshow('img_temp', img_canny)
    lines = cv2.HoughLinesP(img_canny, rho = 1, theta = np.pi/180*60, threshold = 28, minLineLength = 27, maxLineGap = 4)
    if lines is None: return (0,0)
    N = lines.shape[0]
    xmid,ymid = [],[]
    # print(N)
    for i in range(N):
    	x1 = lines[i][0][0]
    	y1 = lines[i][0][1]
    	x2 = lines[i][0][2]
    	y2 = lines[i][0][3]
    	xmid.append((x1+x2) / 2)
    	ymid.append((y1+y2) / 2)
    	# cv2.line(img_temp,(x1,y1),(x2,y2),(255,0,0))
    	# cv2.imshow('img_temp', img_temp)

    #raise NotImplementedError
    center = (int(np.mean(xmid)), int(np.mean(ymid)))
    return center
    raise NotImplementedError

def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """

    img_temp = np.copy(img_in)
    img_red = img_temp[:,:,2]
    img_gre = img_temp[:,:,1]
    img_blu = img_temp[:,:,0]
    img_bi = np.zeros_like(img_red)
    img_bi[(img_red >= 200) & (img_gre >= 200) & (img_blu < 100)] = 255
    img_canny = cv2.Canny(img_bi, 10, 0)
    lines = cv2.HoughLinesP(img_canny, rho = 1, theta = np.pi/180*45, threshold = 20, minLineLength = 40, maxLineGap = 1)
    if lines is None: return (0,0)
    N = lines.shape[0]
    xmid,ymid = [],[]
    # print(N)
    for i in range(N):
    	x1 = lines[i][0][0]
    	y1 = lines[i][0][1]
    	x2 = lines[i][0][2]
    	y2 = lines[i][0][3]
    	xmid.append((x1+x2) / 2)
    	ymid.append((y1+y2) / 2)
    	# cv2.line(img_temp,(x1,y1),(x2,y2),(255,0,0))
    	# cv2.imshow('img_temp', img_temp)
    
    #raise NotImplementedError
    center = (int(np.mean(xmid)), int(np.mean(ymid)))
    return center
    raise NotImplementedError


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """

    img_temp = np.copy(img_in)
    img_red = img_temp[:,:,2]
    img_gre = img_temp[:,:,1]
    img_blu = img_temp[:,:,0]
    img_bi = np.zeros_like(img_red)
    img_bi[(img_red >= 200) & (img_gre >= 100) & (img_gre < 200)] = 255
    # img_bi[(img_gre >= 200) & (img_gre <= 255)] = 0
    # img_bi[(img_blu >= 200) & (img_blu <= 255)] = 0
    img_canny = cv2.Canny(img_bi, 10, 0)
    # cv2.imshow('img_bi',img_bi)
    lines = cv2.HoughLinesP(img_canny, rho = 1, theta = np.pi/180*45, threshold = 20, minLineLength = 40, maxLineGap = 3)
    if lines is None: return (0,0)
    N = lines.shape[0]
    xmid,ymid = [],[]
    # print(img_temp[550,334,:])
    for i in range(N):
    	x1 = lines[i][0][0]
    	y1 = lines[i][0][1]
    	x2 = lines[i][0][2]
    	y2 = lines[i][0][3]
    	xmid.append((x1+x2) / 2)
    	ymid.append((y1+y2) / 2)
    	# cv2.line(img_temp,(x1,y1),(x2,y2),(255,0,0))
    	# cv2.imshow('img_temp', img_temp)
    
    #raise NotImplementedError
    center = (int(np.mean(xmid)), int(np.mean(ymid)))
    return center
    raise NotImplementedError


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """

    img_temp = np.copy(img_in)
    img_red = img_temp[:,:,2]
    img_gre = img_temp[:,:,1]
    img_bi = np.zeros_like(img_red)
    img_bi[(img_red >= 200) & (img_gre <= 100) ] = 255
    img_canny = cv2.Canny(img_bi, 120, 60)
    # cv2.imshow('img_temp', img_canny1)
    circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT,1, 20,
    								param1 = 50,param2 = 30,minRadius = 0,maxRadius = 0)
    # print(circles[0,:])
    if circles is None: return (0,0)
    for circle in circles[0,:]:
    	col = int(circle[0])
    	row = int(circle[1])
    	if img_in[row,col,0] >= 200 and img_in[row,col,2] >= 200 :
    		# cv2.circle(img_temp, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
    		center = (int(circle[0]),int(circle[1]))
    # cv2.imshow('img_temp', img_temp)
    # 	print(img_temp[int(circle[1]), int(circle[0]), :])
    
    return center

    raise NotImplementedError


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    dict = {}

    img_tl = np.copy(img_in)
    radii_range = range(10, 30, 1)
    tl,color = traffic_light_detection(img_tl,radii_range)



    img_ne = np.copy(img_in)
    ne = do_not_enter_sign_detection(img_ne)

    img_st = np.copy(img_in)
    st = stop_sign_detection(img_st)

    img_wn = np.copy(img_in)
    wn = warning_sign_detection(img_wn)

    img_yd = np.copy(img_in)
    yd = yield_sign_detection(img_yd)

    img_cs = np.copy(img_in)
    cs = construction_sign_detection(img_cs)

    if tl != (0,0):
    	dict['traffic_light'] = tl
    if ne != (0,0):
    	dict['no_entry'] = ne
    if st != (0,0):
    	dict['stop'] = st
    if wn != (0,0):
    	dict['warning'] = wn
    if yd != (0,0):
    	dict['yield'] = yd
    if cs != (0,0):
    	dict['construction'] = cs

    return dict

    raise NotImplementedError


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    dict = {}

    img_temp = np.copy(img_in)
    img_pre = cv2.medianBlur(img_temp,5)

    img_tl = np.copy(img_pre)
    img_canny = cv2.Canny(img_tl, 100, 60)
    circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT,1, 25, param1 = 50,param2 = 30,minRadius = 5,maxRadius = 30)
    if circles is None: return ((0,0),'null')
    circle_tl = get_tl(circles)
    if circle_tl is None: return ((0,0),'null')
    for circle in circle_tl:
        col = int(circle[0])
        row = int(circle[1])
        if img_temp[row, col, :][2] == img_temp[row, col, :][1] and img_temp[row, col, :][2] > 250: color = 'yellow'
        elif img_temp[row, col, :][2] > 250 and img_temp[row, col, :][1] < 200: color = 'red'
        elif img_temp[row, col, :][1] > 250 and img_temp[row, col, :][2] < 200: color = 'green'
    tl = (circle_tl[1][0],circle_tl[1][1])


    img_ne = np.copy(img_pre)
    ne = do_not_enter_sign_detection(img_ne)

    img_st = np.copy(img_pre)
    st = stop_sign_detection(img_st)

    img_wn = np.copy(img_pre)
    wn = warning_sign_detection(img_wn)

    img_yd = np.copy(img_pre)
    yd = yield_sign_detection(img_yd)

    img_cs = np.copy(img_pre)
    cs = construction_sign_detection(img_cs)

    if tl != (0,0):
    	dict['traffic_light'] = tl
    if ne != (0,0):
    	dict['no_entry'] = ne
    if st != (0,0):
    	dict['stop'] = st
    if wn != (0,0):
    	dict['warning'] = wn
    if yd != (0,0):
    	dict['yield'] = yd
    if cs != (0,0):
    	dict['construction'] = cs

    return dict

    
    raise NotImplementedError


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    dict = {}
    #---------------------stop sign--------------------------#

    # img_st = np.copy(img_in)
    # img_red = img_st[:,:,2]
    # img_gre = img_st[:,:,1]
    # img_blu = img_st[:,:,0]
    # img_bi = np.zeros_like(img_red)
    # img_bi[(img_red >= 200) & (img_gre <= 200)] = 255
    # img_canny = cv2.Canny(img_bi, 110, 60)
    # # cv2.imshow('img_temp', img_bi)
    # # cv2.waitKey(0)
    # lines = cv2.HoughLinesP(img_canny, rho = 1, theta = np.pi/180*60, threshold = 25, minLineLength = 10, maxLineGap = 1)
    # if lines is None: return (0,0)
    # N = lines.shape[0]
    # xmid,ymid = [],[]
    # # print(N)
    # for i in range(N):
    #     x1 = lines[i][0][0]
    #     y1 = lines[i][0][1]
    #     x2 = lines[i][0][2]
    #     y2 = lines[i][0][3]
    #     xmid.append((x1+x2) / 2)
    #     ymid.append((y1+y2) / 2)
    #     cv2.line(img_st,(x1,y1),(x2,y2),(255,0,0))
    #     cv2.imshow('img_temp', img_st)

    # #raise NotImplementedError
    # st = (int(np.mean(xmid)), int(np.mean(ymid)))
    st = stop_sign_real(img_in)
    if st != (0,0):
        dict['stop'] = st

    #---------------------yield sign--------------------------#
    # img_yd = np.copy(img_in)
    # img_red = img_yd[:,:,2]
    # img_gre = img_yd[:,:,1]
    # img_blu = img_yd[:,:,0]
    # # img_temp = cv2.cvtColor(img_yd, cv2.COLOR_BGR2GRAY)
    # img_bi = np.zeros_like(img_red)
    # img_bi[(img_red >= 200) & (img_gre >= 200) & (img_blu >= 200)] = 255
    # # img_bi[(img_gre >= 200) & (img_gre <= 255)] = 0
    # img_canny = cv2.Canny(img_bi, 110, 60)
    # # cv2.imshow('img_temp', img_bi)
    # lines = cv2.HoughLinesP(img_canny, rho = 1, theta = np.pi/180*30, threshold = 50, minLineLength = 50, maxLineGap = 4)
    # # print(lines)
    # if lines is None: return (0,0)
    # N = lines.shape[0]
    # xmid,ymid = [],[]
    # # print(N)

    # for line in lines:
    #     line_temp = line.flatten()
    #     line_para = Line(line_temp)

    #     if (line_para.angle > 40 and line_para.angle < 80) or (line_para.angle < -40 and line_para.angle > -80) :
    #         x1 = line[0][0]
    #         x2 = line[0][2]
    #         y1 = line[0][1]
    #         y2 = line[0][3]
    #         xmid.append((x1+x2) / 2)
    #         ymid.append((y1+y2) / 2)
    # #         cv2.line(img_temp,(x1,y1),(x2,y2),(255,0,0))
    # # cv2.imshow('img_temp',img_temp)

    # if len(xmid) == 0: return (0,0)
    # yd = (int(np.mean(xmid)), int(np.mean(ymid) - 5))
    yd = yield_sign_real(img_in)
    if yd != (0,0):
        dict['yield'] = yd

    ne = no_enter_real(img_in)
    if ne != (0,0):
        dict['no_entry'] = ne


    return dict

    raise NotImplementedError


def stop_sign_real(img_in):
    img_st = np.copy(img_in)
    img_red = img_st[:,:,2]
    img_gre = img_st[:,:,1]
    img_blu = img_st[:,:,0]
    img_bi = np.zeros_like(img_red)
    img_bi[(img_red >= 150) & (img_gre <= 100)] = 255
    img_canny = cv2.Canny(img_bi, 1100, 500)
    # cv2.imshow('img_temp', img_canny)
    # cv2.waitKey(0)
    lines = cv2.HoughLinesP(img_canny, rho = 1, theta = np.pi/60, threshold = 30, minLineLength = 30, maxLineGap = 8)
    if lines is None: return (0,0)
    N = lines.shape[0]
    xmid,ymid = [],[]
    # print(N)
    for line in lines:
        line_temp = line.flatten()
        line_para = Line(line_temp)
        if (line_para.angle > 85 and line_para.angle < 95) or (line_para.angle > -95 and line_para.angle < -85):
            x1 = line[0][0]
            x2 = line[0][2]
            y1 = line[0][1]
            y2 = line[0][3]
            xmid.append((x1+x2) / 2)
            ymid.append((y1+y2) / 2)
    #         cv2.line(img_st,(x1,y1),(x2,y2),(255,0,0))
    # cv2.imshow('img_temp', img_st)
    if len(xmid) == 0: return (0,0)
    #raise NotImplementedError
    st = (int(np.mean(xmid)), int(np.mean(ymid)))
    # print(st)
    return st

def yield_sign_real(img_in):
    img_yd = np.copy(img_in)
    img_red = img_yd[:,:,2]
    img_gre = img_yd[:,:,1]
    img_blu = img_yd[:,:,0]
    # img_temp = cv2.cvtColor(img_yd, cv2.COLOR_BGR2GRAY)
    img_bi = np.zeros_like(img_red)
    img_bi[(img_red >= 200) & (img_gre <= 100) & (img_blu <= 100)] = 255
    # img_bi[(img_gre >= 200) & (img_gre <= 255)] = 0
    img_canny = cv2.Canny(img_bi, 10, 0)
    # cv2.imshow('img_temp', img_canny)
    lines = cv2.HoughLinesP(img_canny, rho = 1, theta = np.pi/90, threshold = 50, minLineLength = 50, maxLineGap = 7)
    # print(lines)
    if lines is None: return (0,0)
    N = lines.shape[0]
    xmid,ymid = [],[]
    # print(N)

    for line in lines:
        line_temp = line.flatten()
        line_para = Line(line_temp)

        if (line_para.angle > 40 and line_para.angle < 80) or (line_para.angle < -40 and line_para.angle > -80) :
            x1 = line[0][0]
            x2 = line[0][2]
            y1 = line[0][1]
            y2 = line[0][3]
            xmid.append((x1+x2) / 2)
            ymid.append((y1+y2) / 2)
    #         cv2.line(img_yd,(x1,y1),(x2,y2),(0,0,0))
    # cv2.imshow('img_temp',img_yd)

    if len(xmid) == 0: return (0,0)
    xx = np.max(xmid)
    xs = np.min(xmid)
    # yx = np.max(ymid)
    # ys = np.min(ymid)
    yd = (int((xx+xs) / 2), int(np.mean(ymid) - 20))
    return yd


def no_enter_real(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """

    img_ne = np.copy(img_in)
    img_red = img_ne[:,:,2]
    img_gre = img_ne[:,:,1]
    img_bi = np.zeros_like(img_red)
    img_bi[(img_red >= 150) & (img_gre <= 100) ] = 255
    img_canny = cv2.Canny(img_bi, 10, 0)
    # cv2.imshow('img_temp', img_canny)
    # circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT,1, 100,
    #                                 param1 = 100,param2 = 28,minRadius = 50,maxRadius = 100)
    circles = cv2.HoughCircles(img_canny, cv2.HOUGH_GRADIENT,1, 100,
                                    param1 = 100,param2 = 20,minRadius = 25,maxRadius = 50)
    # print(circles[0,:])
    if circles is None: return (0,0)
    for circle in circles[0,:]:
        col = int(circle[0])
        row = int(circle[1])
        if img_in[row,col,0] >= 200 and img_in[row,col,2] >= 200 :
            # cv2.circle(img_ne, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
            ne = (int(circle[0]),int(circle[1]))
    # cv2.imshow('img_temp', img_ne)
            return ne
    return (0,0)
