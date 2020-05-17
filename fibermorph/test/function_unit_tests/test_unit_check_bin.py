import skimage
import numpy as np


def check_bin(img):
    
    img_bool = np.asarray(img, dtype=np.bool)
    
    # Gets the unique values in the image matrix. Since it is binary, there should only be 2.
    unique, counts = np.unique(img_bool, return_counts=True)
    print(unique)
    print("Found this many counts:")
    print(len(counts))
    print(counts)
    
    # If the length of unique is not 2 then print that the image isn't a binary.
    if len(unique) != 2:
        print("Image is not binarized!")
        hair_pixels = len(counts)
        print("There is/are {} value(s) present, but there should be 2!\n".format(hair_pixels))
    # If it is binarized, print out that is is and then get the amount of hair pixels to background pixels.
    if counts[0] < counts[1]:
        print("{} is not reversed".format(str(img)))
        img = skimage.util.invert(img_bool)
        print("Now {} is reversed =)".format(str(img)))
        return img
    
    else:
        print("{} is already reversed".format(str(img)))
        img = img_bool
        
        print(type(img))
        return img

