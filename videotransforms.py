import numpy as np
import numbers
import random

class RandomFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # RGB x t x h x w
            rand_axis = random.randrange(0,3)
            if (rand_axis == 1): rand_axis = 3 # don't flip the time dimension
            return np.flip(imgs.copy(), axis=rand_axis)
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
