"""Cell segmentation from force map for background exclusion."""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, binary_dilation, remove_small_objects, disk
from skimage.measure import label, regionprops


def estimate_cell_mask(heatmap, sigma=2, min_size=200, exclude_full_image=True,
                       threshold_relax=0.85, dilate_radius=4):
    """
    Estimate cell region from force map using Otsu thresholding and morphological cleanup.

    Args:
        heatmap: 2D float array [0, 1] - predicted force map
        sigma: Gaussian smoothing sigma to reduce noise. Default 2.
        min_size: Minimum object size in pixels; smaller objects removed. Default 200.
        exclude_full_image: If True, exclude the largest connected component when it
            covers most of the image (>70%) and use the second largest. Default True.
        threshold_relax: Multiply Otsu threshold by this (<1 = looser, include more pixels).
            Default 0.85.
        dilate_radius: Radius to dilate mask outward to include surrounding pixels.
            Default 4.

    Returns:
        mask: Binary uint8 array, 1 = estimated cell, 0 = background
    """
    heatmap = np.clip(heatmap, 0, 1).astype(np.float64)
    if np.max(heatmap) <= 0:
        return np.zeros_like(heatmap, dtype=np.uint8)

    # Smooth to reduce noise
    smoothed = gaussian_filter(heatmap, sigma=sigma)

    # Otsu automatic threshold, relaxed to include more pixels
    thresh = threshold_otsu(smoothed) * threshold_relax
    mask = (smoothed > thresh).astype(np.uint8)

    # Morphological cleanup
    mask = binary_closing(mask, disk(5)).astype(np.uint8)
    mask = binary_opening(mask, disk(3)).astype(np.uint8)
    mask = remove_small_objects(mask.astype(bool), min_size=min_size).astype(np.uint8)

    # Select component: second largest if largest is whole image
    labeled = label(mask)
    props = list(regionprops(labeled))

    if len(props) == 0:
        return np.zeros_like(heatmap, dtype=np.uint8)

    props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
    total_px = heatmap.shape[0] * heatmap.shape[1]

    if exclude_full_image and len(props_sorted) >= 2 and props_sorted[0].area > 0.7 * total_px:
        region = props_sorted[1]
    else:
        region = props_sorted[0]

    mask = (labeled == region.label).astype(np.uint8)

    # Dilate to include surrounding pixels
    if dilate_radius > 0:
        mask = binary_dilation(mask, disk(dilate_radius)).astype(np.uint8)

    return mask
