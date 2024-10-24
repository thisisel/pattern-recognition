grabbed_s_sample_pixels = read_samples(sky_pic_paths)
stacked_s_sample_pixels = stack_all_samples(grabbed_s_sample_pixels)

print(
    f"total number of pixels gathered from {len(sky_pic_paths)} samples: {stacked_s_sample_pixels.shape[0]}"
)
