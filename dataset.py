"""
Test Dataset class
"""

from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
class testData(Dataset):
	def __init__(self,transforms=None):
		data_dir = '/home/omkar/Documents/Omkar/kaggle/Test/'
		paths = []
		self.paths=[]
		self.transforms = transforms
		count = 1
		for count in range(16111):
			paths.append(os.path.join(data_dir,'Test_' + str(count+1) + '.jpg'))

		self.paths = paths

	def __len__(self):
		return len(self.paths)

	def __getitem__(self,index):
		return self.transforms(Image.open(self.paths[index]))
