import pathlib
import skimage

import numpy as np
from scipy import ndimage

from skimage import io

from fibermorph.test.function_unit_tests.test_unit_remove_particles import remove_particles
from fibermorph.test.function_unit_tests.test_unit_check_bin import check_bin
        
def prune(skeleton, name, pruned_dir, save_img=False):
    """
    Starting with a morphological skeleton, creates a corresponding binary image
    with all branch-points pixels (1) and all other pixels (0).
    """

    print("\nPruning {}...\n".format(name))

    # identify 3-way branch-points through convolving the image using appropriate
    # structure elements for an 8-connected skeleton:
    # http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm
    hit1 = np.array([[0, 1, 0],
                     [0, 1, 0],
                     [1, 0, 1]], dtype=np.uint8)
    hit2 = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [1, 0, 1]], dtype=np.uint8)
    hit3 = np.array([[1, 0, 0],
                     [0, 1, 1],
                     [0, 1, 0]], dtype=np.uint8)
    hit_list = [hit1, hit2, hit3]

    # use some nifty NumPy slicing to add the three remaining rotations of each
    # of the structure elements to the hit list
    for ii in range(9):
        hit_list.append(np.transpose(hit_list[-3])[::-1, ...])

    # add structure elements for branch-points four 4-way branchpoints, these
    hit3 = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], dtype=np.uint8)
    hit4 = np.array([[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]], dtype=np.uint8)
    hit_list.append(hit3)
    hit_list.append(hit4)
    print("Creating hit and miss list")

    # create a zero np.array() of the same shape as the skeleton and use it to collect
    # "hits" from the convolution operation

    skel_image = check_bin(skeleton)
    print("Converting image to binary array")

    branch_points = np.zeros(skel_image.shape)
    print("Creating empty array for branch points")

    for hit in hit_list:
        target = hit.sum()
        curr = ndimage.convolve(skel_image, hit, mode="constant")
        branch_points = np.logical_or(branch_points, np.where(curr == target, 1, 0))

    print("Completed collection of branch points")

    # pixels may "hit" multiple structure elements, ensure the output is a binary
    # image
    branch_points_image = np.where(branch_points, 1, 0)
    print("Ensuring binary")

    # use SciPy's ndimage module to label each contiguous foreground feature
    # uniquely, this will locating and determining coordinates of each branch-point
    labels, num_labels = ndimage.label(branch_points_image)
    print("Labelling branches")

    # use SciPy's ndimage module to determine the coordinates/pixel corresponding
    # to the center of mass of each branchpoint
    branch_points = ndimage.center_of_mass(skel_image, labels=labels, index=range(1, num_labels + 1))
    branch_points = np.array([value for value in branch_points if not np.isnan(value[0]) or not np.isnan(value[1])],
                             dtype=int)
    num_branch_points = len(branch_points)

    hit = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]], dtype=np.uint8)

    dilated_branches = ndimage.convolve(branch_points_image, hit, mode='constant')
    dilated_branches_image = np.where(dilated_branches, 1, 0)
    print("Ensuring binary dilated branches")
    pruned_image = np.subtract(skel_image, dilated_branches_image)
    # pruned_image = np.subtract(skel_image, branch_points_image)
    
    pruned_image = remove_particles(pruned_image, pruned_dir, name, prune=True, save_img=save_img)

    return pruned_image


test_im = skimage.io.imread("/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages/skeleton_curv.tiff", as_gray=True)

test_dir = "/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages"

test_bin = prune(test_im, "pruned_curv", test_dir, save_img=True)

