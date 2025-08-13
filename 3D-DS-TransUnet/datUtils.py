# """
# Jython utilities for cig.
# Author: Xinming Wu, USTC
# Version: 2020.03.28
# """
# from common import *
#
# #############################################################################
# # Internal constants
#
# #_datdir = "/media/xinwu/disk-2/karstFW/"
# _datdir = "../data/"
# _pngdir = "../png/"
#
# #############################################################################
# # Setup
#
# def setupForSubset(name):
#   """
#   Setup for a specified directory includes:
#     seismic directory
#     samplings s1,s2
#   Example: setupForSubset("pnz")
#   """
#   global pngDir
#   global seismicDir
#   global s1,s2,s3
#   global n1,n2,n3
#   if name=="jie":
#     """ jie """
#     print("setupForSubset: jie")
#     pngDir = _pngdir+"jie/"
#     seismicDir = _datdir+"jie/"
#     n1,n2,n3 = 320,1024,1024
#     d1,d2,d3 = 1,1,1 # (s,km/s)
#     f1,f2,f3 = 0,0,0
#     s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
#   elif name=="validation":
#     print ("setupForSubset: validation")
#     pngDir = _pngdir+"validation/"
#     seismicDir = _datdir+"validation/"
#     n1,n2,n3 = 256,256,256
#     d1,d2,d3 = 1.0,1.0,1.0
#     f1,f2,f3 = 0.0,0.0,0.0 # = 0.000,0.000,0.000
#     s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
#   elif name=="hongliu":
#     print ("setupForSubset: hongliu")
#     pngDir = _pngdir+"hongliu/"
#     seismicDir = _datdir+"hongliu/"
#     n1,n2,n3 = 256,256,256
#     d1,d2,d3 = 1.0,1.0,1.0
#     f1,f2,f3 = 0.0,0.0,0.0 # = 0.000,0.000,0.000
#     s1,s2,s3 = Sampling(n1,d1,f1),Sampling(n2,d2,f2),Sampling(n3,d3,f3)
#   else:
#     print("unrecognized subset:",name)
#     System.exit
#
# def getSamplings():
#   return s1,s2,s3
#
# def getSeismicDir():
#   return seismicDir
#
# def getPngDir():
#   return pngDir
#
# #############################################################################
# # read/write images
# def readImageChannels(basename):
#   """
#   Reads three channels of a color image
#   """
#   fileName = seismicDir+basename+".jpg"
#   il = ImageLoader()
#   image = il.readThreeChannels(fileName)
#   return image
# def readColorImage(basename):
#   """
#   Reads three channels of a color image
#   """
#   fileName = seismicDir+basename+".jpg"
#   il = ImageLoader()
#   image = il.readColorImage(fileName)
#   return image
#
# def readImage2D(n1,n2,basename):
#   """
#   Reads an image from a file with specified basename
#   """
#   fileName = seismicDir+basename+".dat"
#   image = zerofloat(n1,n2)
#   ais = ArrayInputStream(fileName)
#   ais.readFloats(image)
#   ais.close()
#   return image
#
# def readImage(basename):
#   """
#   Reads an image from a file with specified basename
#   """
#   fileName = seismicDir+basename+".dat"
#   image = zerofloat(n1,n2,n3)
#   ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
#   ais.readFloats(image)
#   ais.close()
#   return image
#
# def readImage3DB(basename):
#   """
#   Reads an image from a file with specified basename
#   """
#   fileName = seismicDir+basename+".dat"
#   image = zerofloat(n1,n2,n3)
#   ais = ArrayInputStream(fileName,ByteOrder.BIG_ENDIAN)
#   ais.readFloats(image)
#   ais.close()
#   return image
#
# def readImageL(basename):
#   """
#   Reads an image from a file with specified basename
#   """
#   fileName = seismicDir+basename+".dat"
#   image = zerofloat(n1,n2,n3)
#   ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
#   ais.readInts(image)
#   ais.close()
#   return image
#
# def readImage2DL(basename):
#   """
#   Reads an image from a file with specified basename
#   """
#   fileName = seismicDir+basename+".dat"
#   image = zerofloat(n1,n2)
#   ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
#   ais.readFloats(image)
#   ais.close()
#   return image
#
#
# def readImage1D(basename):
#   """
#   Reads an image from a file with specified basename
#   """
#   fileName = seismicDir+basename+".dat"
#   image = zerofloat(n1)
#   ais = ArrayInputStream(fileName)
#   ais.readFloats(image)
#   ais.close()
#   return image
#
# def readImage1L(basename):
#   """
#   Reads an image from a file with specified basename
#   """
#   fileName = seismicDir+basename+".dat"
#   image = zerofloat(n1)
#   ais = ArrayInputStream(fileName,ByteOrder.LITTLE_ENDIAN)
#   ais.readFloats(image)
#   ais.close()
#   return image
#
# def writeImage(basename,image):
#   """
#   Writes an image to a file with specified basename
#   """
#   fileName = seismicDir+basename+".dat"
#   aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
#   aos.writeFloats(image)
#   #aos.writeBytes(image)
#   aos.close()
#   return image
#
# def writeImagex(fname,image):
#   """
#   Writes an image to a file with specified basename
#   """
#   fileName = fname+".dat"
#   aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
#   aos.writeFloats(image)
#   #aos.writeBytes(image)
#   aos.close()
#   return image
#
#
# def writeImageL(basename,image):
#   """
#   Writes an image to a file with specified basename
#   """
#   fileName = seismicDir+basename+".dat"
#   print(fileName)
#   aos = ArrayOutputStream(fileName,ByteOrder.LITTLE_ENDIAN)
#   aos.writeFloats(image)
#   aos.close()
#   return image
#
# #############################################################################
# # read/write fault skins
#
# def skinName(basename,index):
#   return basename+("%05i"%(index))
# def skinIndex(basename,fileName):
#   assert fileName.startswith(basename)
#   i = len(basename)
#   return int(fileName[i:i+5])
#
# def listAllSkinFiles(basename):
#   """ Lists all skins with specified basename, sorted by index. """
#   fileNames = []
#   for fileName in File(seismicDir).list():
#     if fileName.startswith(basename):
#       fileNames.append(fileName)
#   fileNames.sort()
#   return fileNames
#
# def removeAllSkinFiles(basename):
#   """ Removes all skins with specified basename. """
#   fileNames = listAllSkinFiles(basename)
#   for fileName in fileNames:
#     File(seismicDir+fileName).delete()
#
# def readSkin(basename,index):
#   """ Reads one skin with specified basename and index. """
#   return FaultSkin.readFromFile(seismicDir+skinName(basename,index)+".dat")
#
# def getSkinFileNames(basename):
#   """ Reads all skins with specified basename. """
#   fileNames = []
#   for fileName in File(seismicDir).list():
#     if fileName.startswith(basename):
#       fileNames.append(fileName)
#   fileNames.sort()
#   return fileNames
#
# def readSkins(basename):
#   """ Reads all skins with specified basename. """
#   fileNames = []
#   for fileName in File(seismicDir).list():
#     if fileName.startswith(basename):
#       fileNames.append(fileName)
#   fileNames.sort()
#   skins = []
#   for iskin,fileName in enumerate(fileNames):
#     index = skinIndex(basename,fileName)
#     skin = readSkin(basename,index)
#     skins.append(skin)
#   return skins
#
# def writeSkin(basename,index,skin):
#   """ Writes one skin with specified basename and index. """
#   FaultSkin.writeToFile(seismicDir+skinName(basename,index)+".dat",skin)
#
# def writeSkins(basename,skins):
#   """ Writes all skins with specified basename. """
#   for index,skin in enumerate(skins):
#     writeSkin(basename,index,skin)
#
# from org.python.util import PythonObjectInputStream
# def readObject(name):
#   fis = FileInputStream(seismicDir+name+".dat")
#   ois = PythonObjectInputStream(fis)
#   obj = ois.readObject()
#   ois.close()
#   return obj
# def writeObject(name,obj):
#   fos = FileOutputStream(seismicDir+name+".dat")
#   oos = ObjectOutputStream(fos)
#   oos.writeObject(obj)
#   oos.close()

#
# """
# Python utilities for seismic data processing.
# Author: Xinming Wu, USTC
# Version: 2020.03.28
# """
# import os
# import pickle
# import sys
#
# import numpy as np
# import torch
# from matplotlib import pyplot as plt
#
# #############################################################################
# # Internal constants
#
# _datdir = "../data/"
# _pngdir = "../png/"
#
# #############################################################################
# # Setup
#
# class Sampling:
#     pass
#
#
# def setup_for_subset(name):
#     """
#     Setup for a specified directory includes:
#         seismic directory
#         samplings s1,s2,s3
#     Example: setup_for_subset("jie")
#     """
#     global png_dir, seismic_dir, s1, s2, s3, n1, n2, n3
#
#     if name == "jie":
#         print("setup_for_subset: jie")
#         png_dir = os.path.join(_pngdir, "jie/")
#         seismic_dir = os.path.join(_datdir, "jie/")
#         n1, n2, n3 = 320, 1024, 1024
#         d1, d2, d3 = 1, 1, 1
#         f1, f2, f3 = 0, 0, 0
#     elif name == "validation":
#         print("setup_for_subset: validation")
#         png_dir = os.path.join(_pngdir, "validation/")
#         seismic_dir = os.path.join(_datdir, "validation/")
#         n1, n2, n3 = 256, 256, 256
#         d1, d2, d3 = 1.0, 1.0, 1.0
#         f1, f2, f3 = 0.0, 0.0, 0.0
#     elif name == "hongliu":
#         print("setup_for_subset: hongliu")
#         png_dir = os.path.join(_pngdir, "hongliu/")
#         seismic_dir = os.path.join(_datdir, "hongliu/")
#         n1, n2, n3 = 256, 256, 256
#         d1, d2, d3 = 1.0, 1.0, 1.0
#         f1, f2, f3 = 0.0, 0.0, 0.0
#     else:
#         print("unrecognized subset:", name)
#         sys.exit()
#
#     s1, s2, s3 = Sampling(n1, d1, f1), Sampling(n2, d2, f2), Sampling(n3, d3, f3)
#
# def get_samplings():
#     return s1, s2, s3
#
# def get_seismic_dir():
#     return seismic_dir
#
# def get_png_dir():
#     return png_dir
#
# #############################################################################
# # read/write images
#
# def read_image(basename):
#     """
#     Reads an image from a file with specified basename.
#     """
#     file_path = os.path.join(seismic_dir, f"{basename}.dat")
#     image = np.fromfile(file_path, dtype=np.float32)
#     image = image.reshape((n3, n2, n1))
#     return image
#
# def read_image_channels(basename):
#     """
#     Reads three channels of a color image.
#     """
#     file_path = os.path.join(seismic_dir, f"{basename}.jpg")
#     image = plt.imread(file_path)
#     return image
#
# def write_image(basename, image):
#     """
#     Writes an image to a file with specified basename.
#     """
#     file_path = os.path.join(seismic_dir, f"{basename}.dat")
#     image.tofile(file_path)
#
# #############################################################################
# # read/write fault skins
#
# def skin_name(basename, index):
#     return f"{basename}{index:05d}"
#
# def skin_index(basename, file_name):
#     assert file_name.startswith(basename)
#     i = len(basename)
#     return int(file_name[i:i+5])
#
# def list_all_skin_files(basename):
#     """ Lists all skins with specified basename, sorted by index. """
#     file_names = [f for f in os.listdir(seismic_dir) if f.startswith(basename)]
#     file_names.sort()
#     return file_names
#
# def remove_all_skin_files(basename):
#     """ Removes all skins with specified basename. """
#     file_names = list_all_skin_files(basename)
#     for file_name in file_names:
#         os.remove(os.path.join(seismic_dir, file_name))
#
# def read_skin(basename, index):
#     """ Reads one skin with specified basename and index. """
#     file_path = os.path.join(seismic_dir, f"{skin_name(basename, index)}.dat")
#     skin = torch.load(file_path)
#     return skin
#
# def get_skin_file_names(basename):
#     """ Gets all skin file names with specified basename. """
#     return list_all_skin_files(basename)
#
# def read_skins(basename):
#     """ Reads all skins with specified basename. """
#     file_names = list_all_skin_files(basename)
#     skins = [read_skin(basename, skin_index(basename, f)) for f in file_names]
#     return skins
#
# def write_skin(basename, index, skin):
#     """ Writes one skin with specified basename and index. """
#     file_path = os.path.join(seismic_dir, f"{skin_name(basename, index)}.dat")
#     torch.save(skin, file_path)
#
# def write_skins(basename, skins):
#     """ Writes all skins with specified basename. """
#     for index, skin in enumerate(skins):
#         write_skin(basename, index, skin)
#
# def read_object(name):
#     file_path = os.path.join(seismic_dir, f"{name}.dat")
#     with open(file_path, 'rb') as f:
#         obj = pickle.load(f)
#     return obj
#
# def write_object(name, obj):
#     file_path = os.path.join(seismic_dir, f"{name}.dat")
#     with open(file_path, 'wb') as f:
#         pickle.dump(obj, f)
#

# Java库转python
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py

#############################################################################
# Internal constants

_datdir = "../data/"
_pngdir = "../png/"

#############################################################################
# Setup

def setupForSubset(name):
    """
    Setup for a specified directory includes:
      seismic directory
      samplings s1,s2
    Example: setupForSubset("pnz")
    """
    global pngDir
    global seismicDir
    global s1, s2, s3
    global n1, n2, n3
    if name == "jie":
        """ jie """
        print("setupForSubset: jie")
        pngDir = os.path.join(_pngdir, "jie/")
        seismicDir = os.path.join(_datdir, "jie/")
        n1, n2, n3 = 320, 1024, 1024
        d1, d2, d3 = 1, 1, 1  # (s,km/s)
        f1, f2, f3 = 0, 0, 0
        s1, s2, s3 = np.linspace(f1, f1 + (n1 - 1) * d1, n1), np.linspace(f2, f2 + (n2 - 1) * d2, n2), np.linspace(f3, f3 + (n3 - 1) * d3, n3)
    elif name == "validation":
        print("setupForSubset: validation")
        pngDir = os.path.join(_pngdir, "validation/")
        seismicDir = os.path.join(_datdir, "validation/")
        n1, n2, n3 = 256, 256, 256
        d1, d2, d3 = 1.0, 1.0, 1.0
        f1, f2, f3 = 0.0, 0.0, 0.0
        s1, s2, s3 = np.linspace(f1, f1 + (n1 - 1) * d1, n1), np.linspace(f2, f2 + (n2 - 1) * d2, n2), np.linspace(f3, f3 + (n3 - 1) * d3, n3)
    elif name == "hongliu":
        print("setupForSubset: hongliu")
        pngDir = os.path.join(_pngdir, "hongliu/")
        seismicDir = os.path.join(_datdir, "hongliu/")
        n1, n2, n3 = 256, 256, 256
        d1, d2, d3 = 1.0, 1.0, 1.0
        f1, f2, f3 = 0.0, 0.0, 0.0
        s1, s2, s3 = np.linspace(f1, f1 + (n1 - 1) * d1, n1), np.linspace(f2, f2 + (n2 - 1) * d2, n2), np.linspace(f3, f3 + (n3 - 1) * d3, n3)
    else:
        print("unrecognized subset:", name)
        sys.exit(1)

def getSamplings():
    return s1, s2, s3

def getSeismicDir():
    return seismicDir

def getPngDir():
    return pngDir

#############################################################################
# read/write images

def readImageChannels(basename):
    """
    Reads three channels of a color image
    """
    fileName = os.path.join(seismicDir, basename + ".jpg")
    image = plt.imread(fileName)
    return image

def readColorImage(basename):
    """
    Reads three channels of a color image
    """
    return readImageChannels(basename)

def readImage2D(n1, n2, basename):
    """
    Reads an image from a file with specified basename
    """
    fileName = os.path.join(seismicDir, basename + ".dat")
    image = np.fromfile(fileName, dtype=np.float32).reshape(n2, n1)
    return image

def readImage(basename):
    """
    Reads an image from a file with specified basename
    """
    fileName = os.path.join(seismicDir, basename + ".dat")
    image = np.fromfile(fileName, dtype=np.float32).reshape(n3, n2, n1)
    return image

def readImage3DB(basename):
    return readImage(basename)

def readImageL(basename):
    return readImage(basename)

def readImage2DL(basename):
    return readImage2D(n1, n2, basename)

def readImage1D(basename):
    fileName = os.path.join(seismicDir, basename + ".dat")
    image = np.fromfile(fileName, dtype=np.float32).reshape(n1)
    return image

def readImage1L(basename):
    return readImage1D(basename)

def writeImage(basename, image):
    fileName = os.path.join(seismicDir, basename + ".dat")
    image.tofile(fileName)
    return image

def writeImagex(fname, image):
    fileName = fname + ".dat"
    image.tofile(fileName)
    return image

def writeImageL(basename, image):
    fileName = os.path.join(seismicDir, basename + ".dat")
    print(fileName)
    image.tofile(fileName)
    return image

#############################################################################
# read/write fault skins

def skinName(basename, index):
    return basename + ("%05i" % index)

def skinIndex(basename, fileName):
    assert fileName.startswith(basename)
    i = len(basename)
    return int(fileName[i:i+5])

def listAllSkinFiles(basename):
    """ Lists all skins with specified basename, sorted by index. """
    fileNames = [f for f in os.listdir(seismicDir) if f.startswith(basename)]
    fileNames.sort()
    return fileNames

def removeAllSkinFiles(basename):
    """ Removes all skins with specified basename. """
    fileNames = listAllSkinFiles(basename)
    for fileName in fileNames:
        os.remove(os.path.join(seismicDir, fileName))

def readSkin(basename, index):
    """ Reads one skin with specified basename and index. """
    with open(os.path.join(seismicDir, skinName(basename, index) + ".dat"), 'rb') as f:
        skin = pickle.load(f)
    return skin

def getSkinFileNames(basename):
    """ Reads all skins with specified basename. """
    return listAllSkinFiles(basename)

def readSkins(basename):
    """ Reads all skins with specified basename. """
    fileNames = listAllSkinFiles(basename)
    skins = [readSkin(basename, skinIndex(basename, fileName)) for fileName in fileNames]
    return skins

def writeSkin(basename, index, skin):
    """ Writes one skin with specified basename and index. """
    with open(os.path.join(seismicDir, skinName(basename, index) + ".dat"), 'wb') as f:
        pickle.dump(skin, f)

def writeSkins(basename, skins):
    """ Writes all skins with specified basename. """
    for index, skin in enumerate(skins):
        writeSkin(basename, index, skin)

def readObject(name):
    with open(os.path.join(seismicDir, name + ".dat"), 'rb') as f:
        obj = pickle.load(f)
    return obj

def writeObject(name, obj):
    with open(os.path.join(seismicDir, name + ".dat"), 'wb') as f:
        pickle.dump(obj, f)
