from fibermorph import fibermorph
import matplotlib.pyplot as plt


input_file = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_input/section/140918_demo_section.tiff"

input_file2 = "/Users/tinalasisi/Box/01_TPL5158/Box_Dissertation/HairPhenotyping_Methods/data/fibermorph_input/section/ValidationSet_section_TIFF/TIFF/140918_A_1.tiff"

output_dir = "/Users/tinalasisi/Desktop/2020-05-19_fibermorphTest/test_output"

section_df = fibermorph.analyze_section(input_file2, output_dir, resolution=1.06)
