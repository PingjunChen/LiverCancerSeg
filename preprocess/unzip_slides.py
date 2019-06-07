# -*- coding: utf-8 -*-

import os, sys
import zipfile
from pydaily import filesystem


def unzip_slides(slides_dir):
    """ Unzip all slide files
    """

    unzip_dir = os.path.join(os.path.dirname(slides_dir), "LiverImages")
    filesystem.overwrite_dir(unzip_dir)

    zip_list = [ele for ele in os.listdir(slides_dir) if "zip" in ele]
    for ind, ele in enumerate(zip_list):
        print("processing {}/{}".format(ind+1, len(zip_list)))
        zip_ref = zipfile.ZipFile(os.path.join(slides_dir, ele), 'r')
        zip_ref.extractall(unzip_dir)
        zip_ref.close()


if __name__ == "__main__":
    # put all download zipped files here
    source_slides_dir = "../data/SourceData"
    unzip_slides(source_slides_dir)
