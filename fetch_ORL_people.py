from os import listdir, makedirs, remove
from os.path import dirname, join, exists, isdir
from utils import imread, imresize
import numpy as np




class Bunch(dict):
    """Container object for datasets

    Dictionary-like object that exposes its keys as attributes.

    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6

    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass

            
def fetch_ORL_people(data_folder_path):
    
    person_names, file_paths = [], []
    for person_name in listdir(data_folder_path):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path):
            continue
        paths = [join(folder_path, f) for f in listdir(folder_path)]
        n_pictures = len(paths)
        person_name = person_name.replace('_', ' ')
        person_names.extend([person_name] * n_pictures)
        file_paths.extend(paths)

    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)
    faces = load_imgs(file_paths)
    n_faces = len(file_paths)
    
    # shuffle the faces with a deterministic RNG scheme to avoid having
    # all faces of the same person in a row, as it would break some
    # cross validation and learning algorithms such as SGD and online
    # k-means that make an IID assumption
    
    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces, target = faces[indices], target[indices]
    
    X = faces.reshape(len(faces), -1)
    
    return Bunch(data=X, images=faces,
             target=target, target_names=target_names)
        
        
def load_imgs(file_paths, resize=0.5):
    slice_ = (slice(0, 112), slice(0, 92))
    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)


    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)
        
    n_faces = len(file_paths)
    faces = np.zeros((n_faces, h, w), dtype=np.float32)

    # iterate over the collected file path to load the jpeg files as numpy
    # arrays
    for i, file_path in enumerate(file_paths):
        img = imread(file_path)

        face = np.asarray(img[slice_], dtype=np.float32)
        face /= 255.0  # scale uint8 coded colors to the [0.0, 1.0] floats
        if resize is not None:
            face = imresize(face, resize)
        faces[i, ...] = face

    return faces