import sys
import numpy as np

sys.path.append('/opt/watchdog/users/cerussite/ImageSegmentation/SemSeg/semseg-benchmark')

class MockImage:
    def __init__(self, arr, mode='L'):
        self.arr = arr
        self.mode = mode
    def convert(self, mode):
        return MockImage(self.arr, mode)
    def resize(self, size, resample):
        return self
    @classmethod
    def open(cls, path):
        # Return a mock array with some classes including 24, 25, 26, 33
        return cls(np.array([[24, 25], [26, 33]], dtype=np.uint8))
    @classmethod
    def fromarray(cls, arr):
        return cls(arr)

# Mock PIL
import PIL.Image as Image
Image.open = MockImage.open
Image.fromarray = MockImage.fromarray
Image.NEAREST = 0
Image.BILINEAR = 2

# We must also mock np.array so that np.array(MockImage) works.
# Or better just let TF do its thing if we don't mock TF.
