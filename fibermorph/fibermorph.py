#!/usr/bin/env python3
from argparse import ArgumentParser as ap
import os

def parse_args():
    '''
    Parse command-line arguments
    Returns
    -------
    Parser argument namespace
    '''
    parser = ap(description="fibermorph analysis parser")

    parser.add_argument(
        '-i', '--input_directory', metavar='', default=None,
        help='Required. Full path to and name of desired directory containing input files.')

    parser.add_argument(
        '-o', '--output_directory', metavar='', default=None,
        help='Required. Full path to and name of desired output directory. Will be created if it does not exist.')

    parser.add_argument(
        '-j', '--jobs', type=int, metavar='', default=os.cpu_count(),
        help='Integer. Number of parallel jobs to run. Default is the number of cpu cores in the system.')

    parser.add_argument(
        '-simg', '--save_img', type=bool, metavar='', default=True,
        help='Boolean. Defaulted to True.')

    gr_curv = parser.add_argument_group(
        'curvature options', 'arguments used specifically for curvature module')

    gr_curv.add_argument(
        '-cr', '--resolution_mm', type=int, metavar='', default=132,
        help='Integer. Number of pixels per mm for curvature analysis. Default is 132.')

    gr_curv.add_argument(
        '-cws', '--window_size', metavar='', default=None, nargs='+', help='Float or integer or None. Desired size for window of measurement for curvature analysis in pixels or mm (given '
        'the flag --window_unit). If nothing is entered, the default is None and the entire hair will be used to for the curve fitting.')

    gr_curv.add_argument(
        '-cwu', '--window_unit', type=str, default='px', choices=['px', 'mm'], help='String. Unit of measurement for window of measurement for curvature analysis. Can be \'px\' (pixels) or '
        '\'mm\'. Default is \'px\'.')

    gr_curv.add_argument(
        '-cw', '--within_element', action='store_true', default=False, help='Boolean. Default is False. Will create an additional directory with spreadsheets of raw curvature '
        'measurements for each hair if the --within_element flag is included.')

    gr_sect = parser.add_argument_group(
        'section options - arguments used specifically for section analysis module')

    gr_sect.add_argument(
        '-sr', '--resolution_mu', type=float, metavar='', default=4.25,
        help='Float. Number of pixels per micron for section analysis. Default is 4.25.')

    gr_sect.add_argument(
        '-smin', '--minsize', type=int, metavar='', default=20,
        help='Integer. Minimum diameter in microns for sections. Default is 20.')

    gr_sect.add_argument(
        '-smax', '--maxsize', type=int, metavar='', default=150,
        help='Integer. Maximum diameter in microns for sections. Default is 150.')

    gr_opt = parser.add_argument_group(
        'fibermorph module options', 'mutually exclusive modules that can be run with the fibermorph package')

    module_group = gr_opt.add_mutually_exclusive_group(required=True)

    module_group.add_argument(
        '--raw2gray', action='store_true', default=False,
        help='Convert raw image files to grayscale TIFF files.')

    module_group.add_argument(
        '--curvature', action='store_true', default=False,
        help='Analyze curvature in grayscale TIFF images.')

    module_group.add_argument(
        '--section', action='store_true', default=False,
        help='Analyze cross-sections in grayscale TIFF images.')

    module_group.add_argument(
        '--demo_real_curv', action='store_true', default=False,
        help='A demo of fibermorph curvature analysis with real data.')

    module_group.add_argument(
        '--demo_real_section', action='store_true', default=False,
        help='A demo of fibermorph section analysis with real data.')

    # module_group.add_argument(
    #     '--demo_dummy_curv', action='store_true', default=False,
    #     help='A demo of fibermorph curvature with dummy data. Arcs and lines are generated, analyzed and error is '
    #          'calculated.')

    # module_group.add_argument(
    #     '--demo_dummy_section', action='store_true', default=False,
    #     help='A demo of fibermorph section with dummy data. Circles and ellipses are generated, analyzed and error is '
    #          'calculated.')
    return parser

def main():
    # Prase command-line arguments
    parser = parse_args()
    args = parser.parse_args()

    # Run fibermorph
    if args.demo_real_curv:
        from demo import real_curv
        real_curv(args.output_directory)

    if args.demo_real_section:
        pass

    if args.raw2gray:
        pass

    if args.curvature:
        from curvature import Curvature
        curv = Curvature()
        curv.run(args)

    if args.section:
        from section import Section
        sect = Section()
        sect.run(args)

if __name__=='__main__':
    main()
