import os
import torch
import random
import copy
import csv
import medmnist
from medmnist import INFO
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import pydicom as dicom
import cv2
from skimage import transform, io, img_as_float, exposure
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightnessContrast, RandomGamma, OneOf,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, ImageCompression,
    RGBShift, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    OpticalDistortion, RandomResizedCrop, Normalize
)
from albumentations.pytorch import ToTensorV2

def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    # elif mode == "embedding":
    #   transformations_list.append(transforms.Resize((crop_size, crop_size)))
    #   transformations_list.append(transforms.ToTensor())
    #   if normalize is not None:
    #     transformations_list.append(normalize)

    transformSequence = transforms.Compose(transformations_list)

    return transformSequence

def build_ts_transformations():
    AUGMENTATIONS = Compose([
      RandomResizedCrop(size=(224,224)),
      ShiftScaleRotate(rotate_limit=10),
      OneOf([
          RandomBrightnessContrast(),
          RandomGamma(),
           ], p=0.3),
    ])
    return AUGMENTATIONS


class ChestXray14(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14, annotation_percent=2):

    self.img_list = []
    self.img_label = []
 
    self.augment = augment
    self.train_augment = build_ts_transformations()
    

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB').resize((224,224))
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


class CheXpert(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=2):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()
    
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              if self.uncertain_label == "Ones":
                label[i] = 1
              elif self.uncertain_label == "Zeros":
                label[i] = 0
              elif self.uncertain_label == "LSR-Ones":
                label[i] = random.uniform(0.55, 0.85)
              elif self.uncertain_label == "LSR-Zeros":
                label[i] = random.uniform(0, 0.3)
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        self.img_label.append(label)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


class ShenzhenCXR(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=1, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


class VinDrCXR(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=6, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()

    with open(file_path, "r") as fr:
      line = fr.readline().strip()
      while line:
        lineItems = line.split()
        imagePath = os.path.join(images_path, lineItems[0]+".jpeg")
        imageLabel = lineItems[1:]
        imageLabel = [int(i) for i in imageLabel]
        self.img_list.append(imagePath)
        self.img_label.append(imageLabel)
        line = fr.readline()

    if annotation_percent < 100:
      indexes = np.arange(len(self.img_list))
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageLabel = torch.FloatTensor(self.img_label[index])
    imageData = Image.open(imagePath).convert('RGB').resize((224,224))

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


class RSNAPneumonia(Dataset):

  def __init__(self, images_path, file_path, augment, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.strip().split(' ')
          imagePath = os.path.join(images_path, lineItems[0])


          self.img_list.append(imagePath)
          imageLabel = np.zeros(3)
          imageLabel[int(lineItems[-1])] = 1
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])
    
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


class MIMIC(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="Ones", unknown_label=0, annotation_percent=2):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              if self.uncertain_label == "Ones":
                label[i] = 1
              elif self.uncertain_label == "Zeros":
                label[i] = 0
              elif self.uncertain_label == "LSR-Ones":
                label[i] = random.uniform(0.55, 0.85)
              elif self.uncertain_label == "LSR-Zeros":
                label[i] = random.uniform(0, 0.3)
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        self.img_label.append(label)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index): 

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])     

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


## MedMNIST Datasets##

class BaseMNIST(Dataset):
    def __init__(self, images_path, file_path, augment, dataset_name, annotation_percent=100):
        self.dataset_name = dataset_name
        self.image_paths = []
        self.dataset_labels = []
        self.augment = augment
        self.train_augment = build_ts_transformations()
        with open(file_path, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.strip().split(',')
                    imagePath = lineItems[0]
                    imageLabel = ','.join(lineItems[1:]) if len(lineItems) > 2 else lineItems[1]

                    self.image_paths.append(imagePath)
                    self.dataset_labels.append(imageLabel)
                
        self.sample_count = len(self.dataset_labels)
        
        self.indexes = np.arange(self.sample_count)
        if annotation_percent<100:
            random.Random(99).shuffle(self.indexes)
            num_data = int(self.sample_count * annotation_percent / 100.0)
            self.indexes = self.indexes[:num_data]
    
    def __len__(self):
        return len(self.indexes)

class PathMNIST(BaseMNIST):
    def __init__(self, images_path, file_path, augment, annotation_percent=100):
        super().__init__(images_path, file_path, augment, "pathmnist", annotation_percent)
    def __getitem__(self,index):
        
        rawImage = Image.open(self.image_paths[self.indexes[index]])
        imageData  = rawImage.convert('RGB')

        # Convert single-class label to one-hot for multi-class classification
        label_value = int(self.dataset_labels[self.indexes[index]])
        one_hot = torch.zeros(len(INFO[self.dataset_name]["label"]), dtype=torch.float32)
        one_hot[label_value] = 1.0
        imageLabel = one_hot

        if self.augment != None:
            student_img, teacher_img = self.augment(imageData), self.augment(imageData)
        
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image = imageData, mask = imageData)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            student_img=np.array(student_img) / 255.
            teacher_img=np.array(teacher_img) / 255.
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img-mean)/std
            teacher_img = (teacher_img-mean)/std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
            
        return student_img, teacher_img, imageLabel
    


class ChestMNIST(BaseMNIST):
    def __init__(self, images_path, file_path, augment, annotation_percent=100):
        super().__init__(images_path, file_path, augment, "chestmnist", annotation_percent)
    def __getitem__(self,index):
        
        rawImage = Image.open(self.image_paths[self.indexes[index]])
        imageData  = rawImage.convert('RGB')
        imageLabel = torch.FloatTensor([float(x) for x in self.dataset_labels[self.indexes[index]].split(',')])

        if self.augment != None:
            student_img, teacher_img = self.augment(imageData), self.augment(imageData)
        
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image = imageData, mask = imageData)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            student_img=np.array(student_img) / 255.
            teacher_img=np.array(teacher_img) / 255.
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img-mean)/std
            teacher_img = (teacher_img-mean)/std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
            
        return student_img, teacher_img, imageLabel
    
    def __len__(self):
        return len(self.indexes)

class DermaMNIST(BaseMNIST):
    def __init__(self, images_path, file_path, augment, annotation_percent=100):
        super().__init__(images_path, file_path, augment, "dermamnist", annotation_percent)
    
    def __getitem__(self,index):
        
        rawImage = Image.open(self.image_paths[self.indexes[index]])
        imageData  = rawImage.convert('RGB')

        # Convert single-class label to one-hot for multi-class classification
        label_value = int(self.dataset_labels[self.indexes[index]])
        one_hot = torch.zeros(len(INFO[self.dataset_name]["label"]), dtype=torch.float32)
        one_hot[label_value] = 1.0
        imageLabel = one_hot
        
        if self.augment != None:
            student_img, teacher_img = self.augment(imageData), self.augment(imageData)
        
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image = imageData, mask = imageData)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            student_img=np.array(student_img) / 255.
            teacher_img=np.array(teacher_img) / 255.
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img-mean)/std
            teacher_img = (teacher_img-mean)/std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
            
        return student_img, teacher_img, imageLabel


class OCTMNIST(BaseMNIST):
    def __init__(self, images_path, file_path, augment, annotation_percent=100):
        super().__init__(images_path, file_path, augment, "octmnist", annotation_percent)
    def __getitem__(self,index):
        
        rawImage = Image.open(self.image_paths[self.indexes[index]])
        imageData  = rawImage.convert('RGB')

        # Convert single-class label to one-hot for multi-class classification
        label_value = int(self.dataset_labels[self.indexes[index]])
        one_hot = torch.zeros(len(INFO[self.dataset_name]["label"]), dtype=torch.float32)
        one_hot[label_value] = 1.0
        imageLabel = one_hot

        if self.augment != None:
            student_img, teacher_img = self.augment(imageData), self.augment(imageData)
        
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image = imageData, mask = imageData)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            student_img=np.array(student_img) / 255.
            teacher_img=np.array(teacher_img) / 255.
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img-mean)/std
            teacher_img = (teacher_img-mean)/std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
            
        return student_img, teacher_img, imageLabel

class PneumoniaMNIST(BaseMNIST):
    def __init__(self, images_path, file_path, augment, annotation_percent=100):
        super().__init__(images_path, file_path, augment, "pneumoniamnist", annotation_percent)
    def __getitem__(self,index):
        
        rawImage = Image.open(self.image_paths[self.indexes[index]])
        imageData  = rawImage.convert('RGB')

        # Convert to 2-class one-hot
        label_value = int(self.dataset_labels[self.indexes[index]])
        one_hot = torch.zeros(2, dtype=torch.float32)
        one_hot[label_value] = 1.0
        imageLabel = one_hot

        if self.augment != None:
            student_img, teacher_img = self.augment(imageData), self.augment(imageData)
        
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image = imageData, mask = imageData)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            student_img=np.array(student_img) / 255.
            teacher_img=np.array(teacher_img) / 255.
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img-mean)/std
            teacher_img = (teacher_img-mean)/std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
            
        return student_img, teacher_img, imageLabel
    
    def __len__(self):
        return len(self.indexes)

class RetinaMNIST(BaseMNIST):
    def __init__(self, images_path, file_path, augment, annotation_percent=100):
        super().__init__(images_path, file_path, augment, "retinamnist", annotation_percent)
    def __getitem__(self,index):
        
        rawImage = Image.open(self.image_paths[self.indexes[index]])
        imageData  = rawImage.convert('RGB')

        # Convert to one-hot for ordinal classes
        label_value = int(self.dataset_labels[self.indexes[index]])
        one_hot = torch.zeros(len(INFO[self.dataset_name]["label"]), dtype=torch.float32)
        one_hot[label_value] = 1.0
        imageLabel = one_hot

        if self.augment != None:
            student_img, teacher_img = self.augment(imageData), self.augment(imageData)
        
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image = imageData, mask = imageData)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            student_img=np.array(student_img) / 255.
            teacher_img=np.array(teacher_img) / 255.
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img-mean)/std
            teacher_img = (teacher_img-mean)/std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
                  
        return student_img, teacher_img, imageLabel

class BreastMNIST(BaseMNIST):
    def __init__(self, images_path, file_path, augment, annotation_percent=100):
        super().__init__(images_path, file_path, augment, "breastmnist", annotation_percent)
    def __getitem__(self,index):
        
        rawImage = Image.open(self.image_paths[self.indexes[index]])
        imageData  = rawImage.convert('RGB')

        # Convert to 2-class one-hot
        label_value = int(self.dataset_labels[self.indexes[index]])
        one_hot = torch.zeros(2, dtype=torch.float32)
        one_hot[label_value] = 1.0
        imageLabel = one_hot


        if self.augment != None:
            student_img, teacher_img = self.augment(imageData), self.augment(imageData)
        
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image = imageData, mask = imageData)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            student_img=np.array(student_img) / 255.
            teacher_img=np.array(teacher_img) / 255.
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img-mean)/std
            teacher_img = (teacher_img-mean)/std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
            
        return student_img, teacher_img, imageLabel

class BloodMNIST(BaseMNIST):
    def __init__(self, images_path, file_path, augment, annotation_percent=100):
        super().__init__(images_path, file_path, augment, "bloodmnist", annotation_percent)
    def __getitem__(self,index):
        
        rawImage = Image.open(self.image_paths[self.indexes[index]])
        imageData  = rawImage.convert('RGB')

        # Convert single-class label to one-hot for multi-class classification
        label_value = int(self.dataset_labels[self.indexes[index]])
        one_hot = torch.zeros(len(INFO[self.dataset_name]["label"]), dtype=torch.float32)
        one_hot[label_value] = 1.0
        imageLabel = one_hot

        if self.augment != None:
            student_img, teacher_img = self.augment(imageData), self.augment(imageData)
        
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image = imageData, mask = imageData)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            student_img=np.array(student_img) / 255.
            teacher_img=np.array(teacher_img) / 255.
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img-mean)/std
            teacher_img = (teacher_img-mean)/std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
            
        return student_img, teacher_img, imageLabel

class TissueMNIST(BaseMNIST):
    def __init__(self, images_path, file_path, augment, annotation_percent=100):
        super().__init__(images_path, file_path, augment, "tissuemnist", annotation_percent)
    def __getitem__(self,index):
        
        rawImage = Image.open(self.image_paths[self.indexes[index]])
        imageData  = rawImage.convert('RGB')

        # Convert single-class label to one-hot for multi-class classification
        label_value = int(self.dataset_labels[self.indexes[index]])
        one_hot = torch.zeros(len(INFO[self.dataset_name]["label"]), dtype=torch.float32)
        one_hot[label_value] = 1.0
        imageLabel = one_hot

        if self.augment != None:
            student_img, teacher_img = self.augment(imageData), self.augment(imageData)
        
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image = imageData, mask = imageData)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            student_img=np.array(student_img) / 255.
            teacher_img=np.array(teacher_img) / 255.
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img-mean)/std
            teacher_img = (teacher_img-mean)/std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
            
        return student_img, teacher_img, imageLabel

class OrganAMNIST(BaseMNIST):
    def __init__(self, images_path, file_path, augment, annotation_percent=100):
        super().__init__(images_path, file_path, augment, "organamnist", annotation_percent)
    def __getitem__(self,index):
        
        rawImage = Image.open(self.image_paths[self.indexes[index]])
        imageData  = rawImage.convert('RGB')

        # Convert single-class label to one-hot for multi-class classification
        label_value = int(self.dataset_labels[self.indexes[index]])
        one_hot = torch.zeros(len(INFO[self.dataset_name]["label"]), dtype=torch.float32)
        one_hot[label_value] = 1.0
        imageLabel = one_hot

        if self.augment != None:
            student_img, teacher_img = self.augment(imageData), self.augment(imageData)
        
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image = imageData, mask = imageData)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            student_img=np.array(student_img) / 255.
            teacher_img=np.array(teacher_img) / 255.
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img-mean)/std
            teacher_img = (teacher_img-mean)/std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
            
        return student_img, teacher_img, imageLabel

class OrganCMNIST(BaseMNIST):

    def __init__(self, images_path, file_path, augment, annotation_percent=100):
        super().__init__(images_path, file_path, augment, "organcmnist", annotation_percent)
    def __getitem__(self,index):
        
        rawImage = Image.open(self.image_paths[self.indexes[index]])
        imageData  = rawImage.convert('RGB')

        # Convert single-class label to one-hot for multi-class classification
        label_value = int(self.dataset_labels[self.indexes[index]])
        one_hot = torch.zeros(len(INFO[self.dataset_name]["label"]), dtype=torch.float32)
        one_hot[label_value] = 1.0
        imageLabel = one_hot

        if self.augment != None:
            student_img, teacher_img = self.augment(imageData), self.augment(imageData)
        
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image = imageData, mask = imageData)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            student_img=np.array(student_img) / 255.
            teacher_img=np.array(teacher_img) / 255.
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img-mean)/std
            teacher_img = (teacher_img-mean)/std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
            
        return student_img, teacher_img, imageLabel

class OrganSMNIST(BaseMNIST):
    def __init__(self, images_path, file_path, augment, annotation_percent=100):
        super().__init__(images_path, file_path, augment, "organsmnist", annotation_percent)
    def __getitem__(self,index):
        
        rawImage = Image.open(self.image_paths[self.indexes[index]])
        imageData  = rawImage.convert('RGB')

        # Convert single-class label to one-hot for multi-class classification
        label_value = int(self.dataset_labels[self.indexes[index]])
        one_hot = torch.zeros(len(INFO[self.dataset_name]["label"]), dtype=torch.float32)
        one_hot[label_value] = 1.0
        imageLabel = one_hot

        if self.augment != None:
            student_img, teacher_img = self.augment(imageData), self.augment(imageData)
        
        else:
            imageData = (np.array(imageData)).astype('uint8')
            augmented = self.train_augment(image = imageData, mask = imageData)
            student_img = augmented['image']
            teacher_img = augmented['mask']
            student_img=np.array(student_img) / 255.
            teacher_img=np.array(teacher_img) / 255.
            
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            student_img = (student_img-mean)/std
            teacher_img = (teacher_img-mean)/std
            student_img = student_img.transpose(2, 0, 1).astype('float32')
            teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
            
        return student_img, teacher_img, imageLabel

dict_dataloarder = {
    "ChestXray14": ChestXray14,
    "CheXpert": CheXpert,
    "Shenzhen": ShenzhenCXR,
    "VinDrCXR": VinDrCXR,
    "RSNAPneumonia": RSNAPneumonia,
    "MIMIC": MIMIC,
    "PathMNIST": PathMNIST,
    "ChestMNIST": ChestMNIST,
    "DermaMNIST": DermaMNIST,
    "OCTMNIST": OCTMNIST,
    "PneumoniaMNIST": PneumoniaMNIST,
    "RetinaMNIST": RetinaMNIST,
    "BreastMNIST": BreastMNIST,
    "BloodMNIST": BloodMNIST,
    "TissueMNIST": TissueMNIST,
    "OrganAMNIST": OrganAMNIST,
    "OrganCMNIST": OrganCMNIST,
    "OrganSMNIST": OrganSMNIST,
}
