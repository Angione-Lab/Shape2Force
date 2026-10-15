import os

# The bright-field TIFFs carry ImageJ/Fiji metadata tags (50838/50839) that OpenCV does not know and
# reports as warnings on every read. They are harmless; silence them before cv2 is imported anywhere.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
