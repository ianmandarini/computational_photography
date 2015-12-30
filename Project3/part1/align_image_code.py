import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def get_points(im1, im2):
    print('PLEASE, SELECT 2 POINTS IN EACH IMAGE FOR ALIGNMENT.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = np.abs(2*r+1 - R)
    cpad = np.abs(2*c+1 - C)
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale)
    else:
        im2 = sktr.rescale(im2, 1./dscale)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta*180/np.pi)
    return im1, dtheta

def match_img_size(im1, im2, (oh1, ow1), (oh2, ow2)):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[np.floor((h2-h1)/2.) : -np.ceil((h2-h1)/2.), :, :]
    elif h1 > h2:
        im1 = im1[np.floor((h1-h2)/2.) : -np.ceil((h1-h2)/2.), :, :]
    if w1 < w2:
        im2 = im2[:, np.floor((w2-w1)/2.) : -np.ceil((w2-w1)/2.), :]
    elif w1 > w2:
        im1 = im1[:, np.floor((w1-w2)/2.) : -np.ceil((w1-w2)/2.), :]
    assert im1.shape == im2.shape
    return im1, im2

def align_images(im1, im2):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

        oh1 = im1.shape[0]
        oh2 = im2.shape[0]

        ow1 = im1.shape[1]
        ow2 = im2.shape[1]

        pts = get_points(im1, im2)
        im1, im2 = align_image_centers(im1, im2, pts)
        im1, im2 = rescale_images(im1, im2, pts)
        im1, angle = rotate_im1(im1, im2, pts)
        im1, im2 = match_img_size(im1, im2, (oh1, ow1), (oh2, ow2))
        return im1, im2
