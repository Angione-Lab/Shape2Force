"""Cell segmentation from force map for background exclusion."""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, dilation, remove_small_objects, disk
from skimage.measure import label, regionprops


def estimate_cell_mask(heatmap, sigma=2, min_size=200, exclude_full_image=True,
                       threshold_relax=0.85, dilate_radius=4, min_area_ratio=0.2):
    """
    Estimate cell region from force map using Otsu thresholding and morphological cleanup.

    Supports multiple disconnected regions (e.g., two cells): components whose area is
    at least min_area_ratio of the largest are merged into the final mask.

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
        min_area_ratio: Include components with area >= this fraction of the largest
            component (0–1). E.g. 0.2 = include regions at least 20% the size of the
            largest. Handles multiple disconnected force regions. Default 0.2.

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
    mask = closing(mask, disk(5)).astype(np.uint8)
    mask = opening(mask, disk(3)).astype(np.uint8)
    mask = remove_small_objects(mask.astype(bool), min_size=min_size).astype(np.uint8)

    # Select component(s): optionally exclude full-image background, then merge
    # all significant components (handles multiple disconnected force regions)
    labeled = label(mask)
    props = list(regionprops(labeled))

    if len(props) == 0:
        return np.zeros_like(heatmap, dtype=np.uint8)

    props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
    total_px = heatmap.shape[0] * heatmap.shape[1]

    # Skip largest if it covers most of image (likely background)
    if exclude_full_image and len(props_sorted) >= 2 and props_sorted[0].area > 0.7 * total_px:
        props_sorted = props_sorted[1:]

    # Reference area for "significant" components
    ref_area = props_sorted[0].area
    # Include all components with area >= min_area_ratio * ref_area
    labels_to_keep = [p.label for p in props_sorted if p.area >= min_area_ratio * ref_area]

    mask = np.zeros_like(labeled, dtype=np.uint8)
    for lab in labels_to_keep:
        mask[labeled == lab] = 1

    # Dilate to include surrounding pixels
    if dilate_radius > 0:
        mask = dilation(mask, disk(dilate_radius)).astype(np.uint8)

    return mask
