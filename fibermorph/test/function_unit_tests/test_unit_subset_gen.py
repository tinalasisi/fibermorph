def subset_gen(hair_pixel_length, window_size, hair_label):
    subset_start = 0
    if window_size > 10:
        subset_end = int(window_size+subset_start)
    else:
        subset_end = int(hair_pixel_length)
    while subset_end <= hair_pixel_length:
        subset = hair_label[subset_start:subset_end]
        yield subset
        subset_start += 1
        subset_end += 1
