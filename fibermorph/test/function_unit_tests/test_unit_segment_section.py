from fibermorph import fibermorph

import matplotlib.pyplot as plt


img, im_name = fibermorph.imread("/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_input/section/140918_demo_section.tiff")

seg_im = fibermorph.segment_section(img)

plt.imshow(seg_im, cmap='gray')
plt.show()

