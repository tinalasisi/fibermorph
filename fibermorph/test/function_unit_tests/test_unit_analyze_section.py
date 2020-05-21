from fibermorph import fibermorph
import matplotlib.pyplot as plt


input_file = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_input/section/140918_demo_section.tiff"

output_dir = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_output"

section_df = fibermorph.analyze_section(input_file, output_dir, resolution=1.06)
