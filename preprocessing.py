from sr_imports import *
import sr_utils as util
import imghdr
import datetime as dt

def resize(source, destination, size, count=0):
    """
    Resizes images from a source folder.

    Parameters
    ----------
        source : string
                 source folder path
        destination : string
                      destination folder path
        size : int
               resize final dimension
        count : int, optional
                number of images to resize,y 0=ALL.
    """

    valid_extensions = ['png', 'jpg']

    files = os.listdir(source)

    # filter images
    images = filter(lambda f: imghdr.what(f) in valid_extensions, files)
    images = list(images)
    images.sort()

    # shrink images list
    if count > 0 and count <= len(images):
        images = images[:count]

    # resize images
    for i in images:
        i_path = os.path.join(source, i)
        i_proc = Image.open(i_path).resize((size, size))

        s = str(i).split('.')
        i_name = "{0}_{1}.{2}".format(s[0], size, s[1])
        i_proc.save(os.path.join(destination, i_name))