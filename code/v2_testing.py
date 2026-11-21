"""
Random test file to see the version 2 images changes. We can delete this later.
"""

import cv2 as cv
import numpy as np
import config as cf

# Temporary Paths
PATH_TEMP_MTG = 'data/_temp/temp_mtg.dat'
PATH_TEMP_PKM = 'data/_temp/temp_pkm.dat'
PATH_TEMP_YGO = 'data/_temp/temp_ygo.dat'
PATH_TEMP_ALL = 'data/_temp/temp_all.dat'

# Processing Chunk Size
SIZE_CHUNK = 100

# Grey Scale Bounds (0-255) -> (15-240) and more
GREY_LOWER = 15     # Inclusive
GREY_UPPER = 240    # Inclusive
ANGLE_SHIFT = 20
BRIGHTNESS_SHIFT = 40




# Pull the greys scale cards
card = cv.imread(".\data\yugioh\yugioh_000001.jpg", cv.IMREAD_GRAYSCALE)

# Continue if there was an issue
if card is None:
    print(f"[ERROR] Could not process card.")
print(card.shape)
# cv.imshow("Initial Card", card)
# cv.waitKey(0)

# Create a random blurred grey scale background to mimic "real life" backgrounds.
background = np.random.randint(low=GREY_LOWER, high=GREY_UPPER+1, size=(cf.SIZE_PADDED_HEIGHT, cf.SIZE_PADDED_WIDTH), dtype=np.uint8)
cv.GaussianBlur(src=background, dst=background, ksize=(11,11), sigmaX=0, borderType=cv.BORDER_REFLECT)
print(background)
# cv.imshow("Background", background)
# cv.waitKey(0)

# Add in random horizontal perspective tilts (only the card inner part)
# Overlay the card on top of the "real life" background
horizontal_degree_change = np.random.randint(low=-ANGLE_SHIFT, high=ANGLE_SHIFT+1)
vertical_degree_change = np.random.randint(low=-ANGLE_SHIFT, high=ANGLE_SHIFT+1)
# horizontal_degree_change = 0
# vertical_degree_change = -20

# Vertical Perspective Changes
theta_vertical = np.radians(abs(vertical_degree_change))
new_height = cf.SIZE_FEATURE_HEIGHT * np.cos(theta_vertical)
pinch_x = cf.SIZE_FEATURE_WIDTH * np.sin(theta_vertical) * 0.25
offset_y = cf.SIZE_PADDING + (cf.SIZE_FEATURE_HEIGHT - new_height) / 2.0
pinch_top = pinch_x if vertical_degree_change >= 0 else 0.0
pinch_bottom = pinch_x if vertical_degree_change < 0 else 0.0

# Horizontal Perspective Changes
theta_horizontal = np.radians(abs(horizontal_degree_change))
new_width = cf.SIZE_FEATURE_WIDTH * np.cos(theta_horizontal)
pinch_y = cf.SIZE_FEATURE_HEIGHT * np.sin(theta_horizontal) * 0.25
offset_x = cf.SIZE_PADDING + (cf.SIZE_FEATURE_WIDTH - new_width) / 2.0
pinch_right = pinch_y if horizontal_degree_change >= 0 else 0.0
pinch_left = pinch_y if horizontal_degree_change < 0 else 0.0

src_corners = np.array(
    [[0, 0], [cf.SIZE_FEATURE_WIDTH, 0], [cf.SIZE_FEATURE_WIDTH, cf.SIZE_FEATURE_HEIGHT], [0, cf.SIZE_FEATURE_HEIGHT]],
    dtype=np.float32
)

dst_corners = np.array([
    [offset_x + pinch_top, offset_y + pinch_left],
    [offset_x + new_width - pinch_top, offset_y + pinch_right],
    [offset_x + new_width - pinch_bottom, offset_y + new_height - pinch_right],
    [offset_x + pinch_bottom, offset_y + new_height - pinch_left]
], dtype=np.float32)

matrix = cv.getPerspectiveTransform(src=src_corners, dst=dst_corners)
cv.warpPerspective(
    src=card,
    M=matrix,
    dsize=(cf.SIZE_PADDED_WIDTH, cf.SIZE_PADDED_HEIGHT),
    dst=background,
    borderMode=cv.BORDER_TRANSPARENT
)
card = background.astype(np.uint8)
#cv.imshow("Perspective Card", card)
#cv.waitKey(0)

# Randomly shift the brightness of the card: New = |α*Old+β|
brightness_change = np.random.randint(-BRIGHTNESS_SHIFT, BRIGHTNESS_SHIFT)
card = cv.add(src1=card, src2=brightness_change)
cv.imshow("Brightness Card", card)
cv.waitKey(0)


# Flatten the card and normalize it to [0, 1]
card = card.ravel()
#card = cv.normalize(src=card, dst=card_vectors[i], alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

