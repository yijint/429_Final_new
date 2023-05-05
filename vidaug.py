# Copyright (c) 2018 okankop
# The code in this document are taken from https://github.com/okankop/vidaug and sometimes modified slightly

"""
Augmenters that apply transformations to videos.

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.RandomRotate(30),
                          va.RandomResize(0.2)  ])

List of augmenters:
    * RandomRotate
    * RandomTranslate
    * RandomShear
    * HorizontalFlip
    * GaussianBlur
    * ElasticTransformation
    * PiecewiseAffineTransform
    * InvertColor
    * Add
    * Multiply
    * Pepper
    * Salt
    * Sequential
    * OneOf
    * SomeOf
    * Sometimes
"""

import numpy as np
import numbers
import random
import scipy
import skimage
import PIL
import cv2
import PIL.ImageOps as ImageOps

printfunction = False

# Adapted from https://github.com/okankop/vidaug/blob/master/vidaug/augmentors/affine.py :

class RandomRotate(object):
    """
    Rotate video randomly by a random angle within given boundsi.

    Args:
        degrees (sequence or int): Range of degrees to randomly
        select from. If degrees is a number instead of sequence
        like (min, max), the range of degrees, will be
        (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        if printfunction: print("Vidaug: RandomRotate")
        
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        
        return rotated



class RandomTranslate(object):
    """
      Shifting video in X and Y coordinates.

        Args:
            x (int) : Translate in x direction, selected
            randomly from [-x, +x] pixels.

            y (int) : Translate in y direction, selected
            randomly from [-y, +y] pixels.
    """

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __call__(self, clip):
        if printfunction: print("Vidaug: RandomTranslate")
        
        x_move = random.randint(-self.x, +self.x)
        y_move = random.randint(-self.y, +self.y)
        
        if isinstance(clip[0], np.ndarray):
            rows, cols, ch = clip[0].shape
            transform_mat = np.float32([[1, 0, x_move], [0, 1, y_move]])
            return [cv2.warpAffine(img, transform_mat, (cols, rows)) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.transform(img.size, PIL.Image.AFFINE, (1, 0, x_move, 0, 1, y_move)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))


class RandomShear(object):
    """
    Shearing video in X and Y directions.

    Args:
        x (int) : Shear in x direction, selected randomly from
        [-x, +x].

        y (int) : Shear in y direction, selected randomly from
        [-y, +y].
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, clip):
        if printfunction: print("Vidaug: RandomShear")
        
        x_shear = random.uniform(-self.x, self.x)
        y_shear = random.uniform(-self.y, self.y)

        if isinstance(clip[0], np.ndarray):
            rows, cols, ch = clip[0].shape
            transform_mat = np.float32([[1, x_shear, 0], [y_shear, 1, 0]])
            return [cv2.warpAffine(img, transform_mat, (cols, rows)) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.transform(img.size, PIL.Image.AFFINE, (1, x_shear, 0, y_shear, 1, 0)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                'but got list of {0}'.format(type(clip[0])))

# Adapted from https://github.com/okankop/vidaug/blob/master/vidaug/augmentors/intensity.py :

class InvertColor(object):
    """
    Inverts the color of the video.
    """

    def __call__(self, clip):
        if printfunction: print("Vidaug: InvertColor")
        
        if isinstance(clip[0], np.ndarray):
            return [np.invert(img) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            inverted = [ImageOps.invert(img) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return inverted


class Add(object):
    """
    Add a value to all pixel intesities in an video.

    Args:
        value (int): The value to be added to pixel intesities.
    """

    def __init__(self, value=0):
        if value > 255 or value < -255:
            raise TypeError('The video is blacked or whitened out since ' +
                            'value > 255 or value < -255.')
        self.value = value

    def __call__(self, clip):
        if printfunction: print("Vidaug: Add")

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            image = clip[i].astype(np.int32)
            image += self.value
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image = image.astype(np.uint8)
            data_final.append(image.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Multiply(object):
    """
    Multiply all pixel intensities with given value.
    This augmenter can be used to make images lighter or darker.

    Args:
        value (float): The value with which to multiply the pixel intensities
        of video.
    """

    def __init__(self, value=1.0):
        if value < 0.0:
            raise TypeError('The video is blacked out since for value < 0.0')
        self.value = value

    def __call__(self, clip):
        if printfunction: print("Vidaug: Multiply")
        
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            image = clip[i].astype(np.float64)
            image *= self.value
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image = image.astype(np.uint8)
            data_final.append(image.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Pepper(object):
    """
    Augmenter that sets a certain fraction of pixel intensities to 0, hence
    they become black.

    Args:
        ratio (int): Determines number of black pixels on each frame of video.
        Smaller the ratio, higher the number of black pixels.
    """
    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, clip):
        if printfunction: print("Vidaug: Pepper")
        
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            img = clip[i].astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 0, img)
            data_final.append(img.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final

class Salt(object):
    """
    Augmenter that sets a certain fraction of pixel intesities to 255, hence
    they become white.

    Args:
        ratio (int): Determines number of white pixels on each frame of video.
        Smaller the ratio, higher the number of white pixels.
   """
    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, clip):
        if printfunction: print("Vidaug: Salt")
        
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            img = clip[i].astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 255, img)
            data_final.append(img.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final

# Adapted from https://github.com/okankop/vidaug/blob/master/vidaug/augmentors/geometric.py : 

class GaussianBlur(object):
    """
    Augmenter to blur images using gaussian kernels.

    Args:
        sigma (float): Standard deviation of the gaussian kernel.
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, clip):
        if printfunction: print("Vidaug: GaussianBlur")

        if isinstance(clip[0], np.ndarray):
            return [scipy.ndimage.gaussian_filter(img, sigma=self.sigma, order=0) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.filter(PIL.ImageFilter.GaussianBlur(radius=self.sigma)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))


class ElasticTransformation(object):
    """
    Augmenter to transform images by moving pixels locally around using
    displacement fields.
    See
        Simard, Steinkraus and Platt
        Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis
        in Proc. of the International Conference on Document Analysis and
        Recognition, 2003
    for a detailed explanation.

    Args:
        alpha (float): Strength of the distortion field. Higher values mean
        more "movement" of pixels.

        sigma (float): Standard deviation of the gaussian kernel used to
        smooth the distortion fields.

        order (int): Interpolation order to use. Same meaning as in
        `scipy.ndimage.map_coordinates` and may take any integer value in
        the range 0 to 5, where orders close to 0 are faster.

        cval (int): The constant intensity value used to fill in new pixels.
        This value is only used if `mode` is set to "constant".
        For standard uint8 images (value range 0-255), this value may also
        come from the range 0-255. It may be a float value, even for
        integer image dtypes.

        mode : Parameter that defines the handling of newly created pixels.
        May take the same values as in `scipy.ndimage.map_coordinates`,
        i.e. "constant", "nearest", "reflect" or "wrap".
    """
    def __init__(self, alpha=0, sigma=0, order=3, cval=0, mode="constant",
                 name=None, deterministic=False):
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.cval = cval
        self.mode = mode

    def __call__(self, clip):
        if printfunction: 
            print(f"Vidaug: ElasticTransformation with alpha = {self.alpha} and cval = {self.cval}")

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        result = []
        nb_images = len(clip)
        for i in range(nb_images):
            image = clip[i]
            image_first_channel = np.squeeze(image[..., 0])
            indices_x, indices_y = self._generate_indices(image_first_channel.shape, alpha=self.alpha, sigma=self.sigma)
            result.append(self._map_coordinates(
                clip[i],
                indices_x,
                indices_y,
                order=self.order,
                cval=self.cval,
                mode=self.mode))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in result]
        else:
            return result

    def _generate_indices(self, shape, alpha, sigma):
        assert (len(shape) == 2),"shape: Should be of size 2!"
        dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    def _map_coordinates(self, image, indices_x, indices_y, order=1, cval=0, mode="constant"):
        assert (len(image.shape) == 3),"image.shape: Should be of size 3!"
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in range(image.shape[2]):
            remapped_flat = scipy.ndimage.interpolation.map_coordinates(
                image[..., c],
                (indices_x, indices_y),
                order=order,
                cval=cval,
                mode=mode
            )
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result



class PiecewiseAffineTransform(object):
    """
    Augmenter that places a regular grid of points on an image and randomly
    moves the neighbourhood of these point around via affine transformations.

     Args:
         displacement (init): gives distorted image depending on the valuse of displacement_magnification and displacement_kernel

         displacement_kernel (init): gives the blury effect

         displacement_magnification (float): it magnify the image
    """
    def __init__(self, displacement=0, displacement_kernel=0, displacement_magnification=0):
        self.displacement = displacement
        self.displacement_kernel = displacement_kernel
        self.displacement_magnification = displacement_magnification

    def __call__(self, clip):
        if printfunction: print("Vidaug: PiecewiseAffineTransform")

        ret_img_group = clip
        if isinstance(clip[0], np.ndarray):
            im_size = clip[0].shape
            image_w, image_h = im_size[1], im_size[0]
        elif isinstance(clip[0], PIL.Image.Image):
            im_size = clip[0].size
            image_w, image_h = im_size[0], im_size[1]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        displacement_map = np.random.rand(image_h, image_w, 2) * 2 * self.displacement - self.displacement
        displacement_map = cv2.GaussianBlur(displacement_map, None,
                                            self.displacement_kernel)
        displacement_map *= self.displacement_magnification * self.displacement_kernel
        displacement_map = np.floor(displacement_map).astype('int32')

        displacement_map_rows = displacement_map[..., 0] + np.tile(np.arange(image_h), (image_w, 1)).T.astype('int32')
        displacement_map_rows = np.clip(displacement_map_rows, 0, image_h - 1)

        displacement_map_cols = displacement_map[..., 1] + np.tile(np.arange(image_w), (image_h, 1)).astype('int32')
        displacement_map_cols = np.clip(displacement_map_cols, 0, image_w - 1)

        if isinstance(clip[0], np.ndarray):
            return [img[(displacement_map_rows.flatten(), displacement_map_cols.flatten())].reshape(img.shape) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [PIL.Image.fromarray(np.asarray(img)[(displacement_map_rows.flatten(), displacement_map_cols.flatten())].reshape(np.asarray(img).shape)) for img in clip]

# Adapted from https://github.com/okankop/vidaug/blob/master/vidaug/augmentors/flip.py :

class HorizontalFlip(object):
    """
    Horizontally flip the video.
    """

    def __call__(self, clip):
        if printfunction: print("Vidaug: HorizontalFlip")
        
        if isinstance(clip[0], np.ndarray):
            return [np.fliplr(img) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(clip[0])))

# Adapted from https://github.com/okankop/vidaug/blob/master/vidaug/augmentors/group.py :

class Sequential(object):
    """
    Composes several augmentations together.

    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.

        random_order (bool): Whether to apply the augmentations in random order.
    """

    def __init__(self, transforms, random_order=False):
        self.transforms = transforms
        self.rand = random_order

    def __call__(self, clip):
        if self.rand:
            rand_transforms = self.transforms[:]
            random.shuffle(rand_transforms)
            for t in rand_transforms:
                clip = t(clip)
        else:
            for t in self.transforms:
                clip = t(clip)

        return clip


class OneOf(object):
    """
    Selects one augmentation from a list.

    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        select = random.choice(self.transforms)
        clip = select(clip)
        return clip


class SomeOf(object):
    """
    Selects a given number of augmentation from a list.

    Args:
        transforms (list of "Augmentor" objects): The list of augmentations.

        N (int): The number of augmentations to select from the list.

        random_order (bool): Whether to apply the augmentations in random order.

    """

    def __init__(self, transforms, N, random_order=True):
        self.transforms = transforms
        self.rand = random_order
        if N > len(transforms):
            raise TypeError('The number of applied augmentors should be smaller than the given augmentation number')
        else:
            self.N = N

    def __call__(self, clip):
        if self.rand:
            tmp = self.transforms[:]
            selected_trans = [tmp.pop(random.randrange(len(tmp))) for _ in range(self.N)]
            for t in selected_trans:
                clip = t(clip)
            return clip
        else:
            indices = [i for i in range(len(self.transforms))]
            selected_indices = [indices.pop(random.randrange(len(indices)))
                                for _ in range(self.N)]
            selected_indices.sort()
            selected_trans = [self.transforms[i] for i in selected_indices]
            for t in selected_trans:
                clip = t(clip)
            return clip


class Sometimes(object):
    """
    Applies an augmentation with a given probability.

    Args:
        p (float): The probability to apply the augmentation.

        transform (an "Augmentor" object): The augmentation to apply.

    Example: Use this this transform as follows:
        sometimes = lambda aug: va.Sometimes(0.5, aug)
        sometimes(va.HorizontalFlip)
    """

    def __init__(self, p, transform):
        self.transform = transform
        if (p > 1.0) | (p < 0.0):
            raise TypeError('Expected p to be in [0.0 <= 1.0], ' +
                            'but got p = {0}'.format(p))
        else:
            self.p = p

    def __call__(self, clip):
        if random.random() < self.p:
            clip = self.transform(clip)
        return clip
