import skimage
from skimage import io

from fibermorph.fibermorph import analyze_all_curv

# analyzes curvature for entire image (analyze_each does each hair in image)


input_file = skimage.io.imread("/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages/pruned_curv.tiff", as_gray=True)

output_path = "/Users/tinalasisi/Desktop/2019_05_17_fibermorphTestImages"

sorted_df = analyze_all_curv(input_file, "testcase", output_path, resolution=132, window_size_mm=1)