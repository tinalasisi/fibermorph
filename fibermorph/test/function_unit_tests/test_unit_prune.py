import skimage


from skimage import io

from fibermorph.fibermorph import prune



test_im = skimage.io.imread("/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages/skeleton_curv.tiff", as_gray=True)

test_dir = "/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages"

test_bin = prune(test_im, "pruned_curv", test_dir, save_img=True)

