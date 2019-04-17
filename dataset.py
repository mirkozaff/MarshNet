import torch
from torch.utils import data
import os
from skimage import io, transform
import numpy as np
import PIL

'''
============================
DATASET INFO:
===========================
img_width = 163
img_height = 142
==========================
'''

class Dataset(data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, data, labels, transform = None):
            'Initialization'
            self.labels = labels
            self.data = data
            self.transform = transform

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.data)

      def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            img_name = self.data[index]

            # Load data and get label            
            y = self.labels[index]

            if y:
                  img_path = os.path.join('./cell_images/Parasitized', img_name)

            else:
                  img_path = os.path.join('./cell_images/Uninfected', img_name)
                  
            image = PIL.Image.open(img_path)

            if self.transform is not None:
                  image = self.transform(image).double()
                  
            return image, y

class Importer():
      def __init__(self, path = ""):
            self.path = path
            self.data = []
            self.label = []

      def data_importer(self):
            for img in subfile_list('./cell_images/Uninfected', '.png'):
                  self.data.append(img)
                  self.label.append(0)
            for img in subfile_list('./cell_images/Parasitized', '.png'):
                  self.data.append(img)
                  self.label.append(1)
            return self.data, self.label

def subdir_list(dir):
	"""
	Get a list of all sub directory in a directory. It is not recursive.
	Parameters
	----------
	dir: string
		the directory of interest
	Returns
	-------
	subdir : list
		list of all sub directory in a directory
	"""
	return [subdir for subdir in os.listdir(dir) if os.path.isdir(os.path.join(dir, subdir))]

def subfile_list(dir, ext = None):
      """
      Get a list of all file in a directory. It is not recursive.
      Parameters
      ----------
      dir: string
            the directory of interest
      ext: string
            the extension of interest (optional)
      Returns
      -------
      subdir : list
            list of all files in a directory
      """
      if ext is not None:
	      return [subdir for subdir in os.listdir(dir) if (not os.path.isdir(os.path.join(dir, subdir))) and subdir.endswith(ext)]
      else:
            return [subdir for subdir in os.listdir(dir) if not os.path.isdir(os.path.join(dir, subdir))]