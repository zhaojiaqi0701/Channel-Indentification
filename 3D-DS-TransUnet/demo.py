# # #############################################################################
# # """
# # Demo of demostrating fault results
# # Author: Xinming Wu, USTC
# # Version: 2020.03.26
# # """
# # from keras.src.ops import copy
# #
# # from datUtils import *
# # #############################################################################
# #
# # plotOnly = True
# # plotOnly = False
# #
# # def main(args):
# #   if args[1]=="jie":
# #     goJie()
# #   elif args[1]=="hongliu":
# #     goHongliu()
# #   elif args[1]=="valid":
# #     goValid(args[2])
# #   else:
# #     print("demo not found")
# #
# # def goValid(fname):
# #   setupForSubset("validation")
# #   global pngDir
# #   pngDir = getPngDir()
# #   s1,s2,s3 = getSamplings()
# #   gx = readImage("nx/"+fname)
# #   fp = readImage("px/"+fname)
# #   ks = [220,212,50]
# #   vt=[-0.0,-0.0,0.0]
# #   ae=[125,20]
# #   plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=ae,vt=vt,clab="Amplitude",png="gx")
# #   plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=ae,vt=vt,cmap=jetRamp(1.0),
# #           clab="Karst probability",png="fp")
# #
# # def goJie():
# #   gxfile = "gg" # input seismic image
# #   fpfile = "fp" # karst probability by cnn
# #   setupForSubset("jie")
# #   global pngDir
# #   pngDir = getPngDir()
# #   s1,s2,s3 = getSamplings()
# #   gx = readImage(gxfile)
# #   fp = readImage(fpfile)
# #   ks = [168,345,752]
# #   vt=[-0.25,-0.48,0.0]
# #   ae=[-45,50]
# #   plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=ae,vt=vt,clab="Amplitude",png="gx")
# #   plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=ae,vt=vt,cmap=jetRamp(1.0),
# #           clab="Karst probability",png="fp")
# # def goHongliu():
# #   gxfile = "gg" # input seismic image
# #   fpfile = "fp" # karst probability by cnn
# #   setupForSubset("hongliu")
# #   global pngDir
# #   pngDir = getPngDir()
# #   s1,s2,s3 = getSamplings()
# #   gx = readImage(gxfile)
# #   fp = readImage(fpfile)
# #   ks = [174,195,60]
# #   ae = [135,35]
# #   vt = [-0.0,-0.0,0.0]
# #   plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=ae,vt=vt,clab="Amplitude",png="gx")
# #   plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=ae,vt=vt,cmap=jetRamp(1.0),
# #           clab="Karst probability",png="fp")
# #
# # def gain(x):
# #   g = mul(x,x)
# #   ref = RecursiveExponentialFilter(20.0)
# #   ref.apply1(g,g)
# #   div(x,sqrt(g),x)
# #   return x
# #
# # def checkNaN(gx):
# #   n3 = len(gx)
# #   n2 = len(gx[0])
# #   n1 = len(gx[0][0])
# #   for i3 in range(n3):
# #     for i2 in range(n2):
# #       for i1 in range(n1):
# #         if(gx[i3][i2][i1]!=gx[i3][i2][i1]):
# #           gx[i3][i2][i1] = 0
# #   return gx
# #
# # def smooth(sig,u):
# #   v = copy(u)
# #   rgf = RecursiveGaussianFilterP(sig)
# #   rgf.apply0(u,v)
# #   return v
# #
# # def smooth2(sig1,sig2,u):
# #   v = copy(u)
# #   rgf1 = RecursiveGaussianFilterP(sig1)
# #   rgf2 = RecursiveGaussianFilterP(sig2)
# #   rgf1.apply0X(u,v)
# #   rgf2.applyX0(v,v)
# #   return v
# #
# # def normalize(e):
# #   emin = min(e)
# #   emax = max(e)
# #   return mul(sub(e,emin),1.0/(emax-emin))
# #
# # def slice12(k3,f):
# #   n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
# #   s = zerofloat(n1,n2)
# #   SimpleFloat3(f).get12(n1,n2,0,0,k3,s)
# #   return s
# #
# # def slice13(k2,f):
# #   n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
# #   s = zerofloat(n1,n3)
# #   SimpleFloat3(f).get13(n1,n3,0,k2,0,s)
# #   return s
# #
# # def slice23(k1,f):
# #   n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
# #   s = zerofloat(n2,n3)
# #   SimpleFloat3(f).get23(n2,n3,k1,0,0,s)
# #   return s
# #
# # #############################################################################
# # # graphics
# #
# # def jetFill(alpha):
# #   return ColorMap.setAlpha(ColorMap.JET,alpha)
# # def jetFillExceptMin(alpha):
# #   a = fillfloat(alpha,256)
# #   a[0] = 0.0
# #   return ColorMap.setAlpha(ColorMap.JET,a)
# # def jetRamp(alpha):
# #   return ColorMap.setAlpha(ColorMap.JET,rampfloat(0.0,alpha/256,256))
# # def bwrFill(alpha):
# #   return ColorMap.setAlpha(ColorMap.BLUE_WHITE_RED,alpha)
# # def bwrNotch(alpha):
# #   a = zerofloat(256)
# #   for i in range(len(a)):
# #     if i<128:
# #       a[i] = alpha*(128.0-i)/128.0
# #     else:
# #       a[i] = alpha*(i-127.0)/128.0
# #   return ColorMap.setAlpha(ColorMap.BLUE_WHITE_RED,a)
# # def hueFill(alpha):
# #   return ColorMap.getHue(0.0,1.0,alpha)
# # def hueFillExceptMin(alpha):
# #   a = fillfloat(alpha,256)
# #   a[0] = 0.0
# #   return ColorMap.setAlpha(ColorMap.getHue(0.0,1.0),a)
# #
# # def addColorBar(frame,clab=None,cint=None):
# #   cbar = ColorBar(clab)
# #   if cint:
# #     cbar.setInterval(cint)
# #   cbar.setFont(Font("Arial",Font.PLAIN,32)) # size by experimenting
# #   cbar.setWidthMinimum
# #   cbar.setBackground(Color.WHITE)
# #   frame.add(cbar,BorderLayout.EAST)
# #   return cbar
# #
# # def convertDips(ft):
# #   return FaultScanner.convertDips(0.2,ft) # 5:1 vertical exaggeration
# #
# # def plot2(s1,s2,x,u=None,g=None,x1=None,c=None,
# #         cmap=ColorMap.GRAY,clab="Amplitude",
# #         cmin=-2,cmax=2,title=None,png=None):
# #   sp = SimplePlot(SimplePlot.Origin.UPPER_LEFT)
# #   if title:
# #     sp.setTitle(title)
# #   n1,n2=s1.count,s2.count
# #   sp.addColorBar(clab)
# #   #sp.setSize(955,400)
# #   sp.setSize(755,500)
# #   sp.setHLabel("Inline (sample)")
# #   sp.setVLabel("Depth (sample)")
# #   sp.plotPanel.setColorBarWidthMinimum(60)
# #   sp.setVLimits(0,n1-1)
# #   sp.setHLimits(0,n2-1)
# #   sp.setFontSize(16)
# #   pv = sp.addPixels(s1,s2,x)
# #   pv.setColorModel(cmap)
# #   pv.setInterpolation(PixelsView.Interpolation.LINEAR)
# #   if cmin<cmax:
# #     pv.setClips(cmin,cmax)
# #   if u:
# #     cv = sp.addContours(s1,s2,u)
# #     cv.setContours(80)
# #     cv.setLineColor(Color.YELLOW)
# #   if g:
# #     pv = sp.addPixels(s1,s2,g)
# #     pv.setInterpolation(PixelsView.Interpolation.NEAREST)
# #     pv.setColorModel(ColorMap.getJet(0.8))
# #     pv.setClips(0.1,s1.count)
# #   if x1:
# #     x1k = zerofloat(n2)
# #     x2  = zerofloat(n2)
# #     x1s  = zerofloat(n1)
# #     for i1 in range(n1):
# #       x1s[i1] = i1
# #     cp = ColorMap(0,n1,ColorMap.JET)
# #     rgb = cp.getRgbFloats(x1s)
# #     ref = RecursiveExponentialFilter(1)
# #     for k in range(20,n1-20,15):
# #       for i2 in range(n2):
# #         x2[i2] = i2
# #         x1k[i2] = x1[i2][k]
# #       ref.apply(x1k,x1k)
# #       pv = sp.addPoints(x1k,x2)
# #       pv.setLineWidth(2.5)
# #       r,g,b=rgb[k*3],rgb[k*3+1],rgb[k*3+2]
# #       pv.setLineColor(Color(r,g,b))
# #   if pngDir and png:
# #     sp.paintToPng(700,3.333,pngDir+png+".png")
# #
# # def plot3(s1,s2,s3,f,g=None,cmin=-2,cmax=2,zs=0.5,sc=1.4,
# #         ks=[175,330,377],ae=[45,35],vt=[-0.1,-0.06,0.0],
# #         cmap=None,clab=None,cint=None,surf=None,png=None):
# #   n3 = len(f)
# #   n2 = len(f[0])
# #   n1 = len(f[0][0])
# #   d1,d2,d3 = s1.delta,s2.delta,s3.delta
# #   f1,f2,f3 = s1.first,s2.first,s3.first
# #   l1,l2,l3 = s1.last,s2.last,s3.last
# #   sf = SimpleFrame(AxesOrientation.XRIGHT_YOUT_ZDOWN)
# #   cbar = None
# #   if g==None:
# #     ipg = sf.addImagePanels(s1,s2,s3,f)
# #     if cmap!=None:
# #       ipg.setColorModel(cmap)
# #     if cmin!=None and cmax!=None:
# #       ipg.setClips(cmin,cmax)
# #     else:
# #       ipg.setClips(-2.0,2.0)
# #     if clab:
# #       cbar = addColorBar(sf,clab,cint)
# #       ipg.addColorMapListener(cbar)
# #   else:
# #     ipg = ImagePanelGroup2(s1,s2,s3,f,g)
# #     ipg.setClips1(-2,2)
# #     if cmin!=None and cmax!=None:
# #       ipg.setClips2(cmin,cmax)
# #     if cmap==None:
# #       cmap = jetFill(0.8)
# #     ipg.setColorModel2(cmap)
# #     if clab:
# #       cbar = addColorBar(sf,clab,cint)
# #       ipg.addColorMap2Listener(cbar)
# #     sf.world.addChild(ipg)
# #   if cbar:
# #     cbar.setWidthMinimum(120)
# #   if surf:
# #     tg = TriangleGroup(True,surf)
# #     sf.world.addChild(tg)
# #   ipg.setSlices(ks[0],ks[1],ks[2])
# #   '''
# #   if cbar:
# #     sf.setSize(987,720)
# #   else:
# #     sf.setSize(850,720)
# #   '''
# #   if cbar:
# #     sf.setSize(887,750)
# #   else:
# #     sf.setSize(750,750)
# #
# #   vc = sf.getViewCanvas()
# #   vc.setBackground(Color.WHITE)
# #   radius = 0.5*sqrt(n1*n1+n2*n2+n3*n3)
# #   ov = sf.getOrbitView()
# #   zscale = zs*max(n2*d2,n3*d3)/(n1*d1)
# #   #ov.setAxesScale(1.0,1.0,zscale)
# #   ov.setScale(sc)
# #   ov.setWorldSphere(BoundingSphere(0.5*n1,0.5*n2,0.5*n3,radius))
# #   #ov.setWorldSphere(BoundingSphere(BoundingBox(f3,f2,f1,l3,l2,l1)))
# #   ov.setTranslate(Vector3(vt[0],vt[1],vt[2]))
# #   ov.setAzimuthAndElevation(ae[0],ae[1])
# #   sf.setVisible(True)
# #   if png and pngDir:
# #     sf.paintToFile(pngDir+png+".png")
# #     if cbar:
# #       cbar.paintToPng(720,1,pngDir+png+"cbar.png")
# # #############################################################################
# # # Run the function main on the Swing thread
# # import sys
# # class _RunMain(Runnable):
# #   def __init__(self,main):
# #     self.main = main
# #   def run(self):
# #     self.main(sys.argv)
# # def run(main):
# #   SwingUtilities.invokeLater(_RunMain(main))
# # run(main)
# #


# #############################################################################
# """
# Demo of demonstrating fault results
# Author: Xinming Wu, USTC
# Version: 2020.03.26
# """
# import os
# import sys
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from matplotlib.cm import ScalarMappable
# #
# # from datUtils import *
# # #############################################################################
# #
# # plotOnly = True
# # plotOnly = False
# #
# # def main(args):
# #     if args[1] == "jie":
# #         go_jie()
# #     elif args[1] == "hongliu":
# #         go_hongliu()
# #     elif args[1] == "valid":
# #         go_valid(args[2])
# #     else:
# #         print("demo not found")
# #
# # def load_model(mk):
# #     global model
# #     model = torch.load(f'check/checkpoint.{mk}.pth')
# #     model.eval()
# #
# # def read_image(path):
# #     image = np.fromfile(path, dtype=np.float32)
# #     image = image.reshape((320, 1024, 1024))
# #     return image
# #
# # def go_valid(fname):
# #     setup_for_subset("validation")
# #     global png_dir
# #     png_dir = get_png_dir()
# #     s1, s2, s3 = get_samplings()
# #     gx = read_image(f"nx/{fname}.dat")
# #     fp = read_image(f"px/{fname}.dat")
# #     ks = [220, 212, 50]
# #     vt = [-0.0, -0.0, 0.0]
# #     ae = [125, 20]
# #     plot3(s1, s2, s3, gx, cmin=-2, cmax=2, ks=ks, ae=ae, vt=vt, clab="Amplitude", png="gx")
# #     plot3(s1, s2, s3, gx, fp, cmin=0.2, cmax=1, ks=ks, ae=ae, vt=vt, cmap='jet',
# #           clab="Karst probability", png="fp")
# #
# # def go_jie():
# #     gxfile = "gg"  # input seismic image
# #     fpfile = "fp"  # karst probability by cnn
# #     setup_for_subset("jie")
# #     global png_dir
# #     png_dir = get_png_dir()
# #     s1, s2, s3 = get_samplings()
# #     gx = read_image(gxfile)
# #     fp = read_image(fpfile)
# #     ks = [168, 345, 752]
# #     vt = [-0.25, -0.48, 0.0]
# #     ae = [-45, 50]
# #     plot3(s1, s2, s3, gx, cmin=-2, cmax=2, ks=ks, ae=ae, vt=vt, clab="Amplitude", png="gx")
# #     plot3(s1, s2, s3, gx, fp, cmin=0.2, cmax=1, ks=ks, ae=ae, vt=vt, cmap='jet',
# #           clab="Karst probability", png="fp")
# #
# # def go_hongliu():
# #     gxfile = "gg"  # input seismic image
# #     fpfile = "fp"  # karst probability by cnn
# #     setup_for_subset("hongliu")
# #     global png_dir
# #     png_dir = get_png_dir()
# #     s1, s2, s3 = get_samplings()
# #     gx = read_image(gxfile)
# #     fp = read_image(fpfile)
# #     ks = [174, 195, 60]
# #     ae = [135, 35]
# #     vt = [-0.0, -0.0, 0.0]
# #     plot3(s1, s2, s3, gx, cmin=-2, cmax=2, ks=ks, ae=ae, vt=vt, clab="Amplitude", png="gx")
# #     plot3(s1, s2, s3, gx, fp, cmin=0.2, cmax=1, ks=ks, ae=ae, vt=vt, cmap='jet',
# #           clab="Karst probability", png="fp")
# #
# # def plot3(s1, s2, s3, gx, fp=None, cmin=-2, cmax=2, ks=[175, 330, 377], ae=[45, 35], vt=[-0.1, -0.06, 0.0], cmap='gray', clab=None, png=None):
# #     fig = plt.figure(figsize=(15, 10))
# #     ax = fig.add_subplot(131)
# #     im = ax.imshow(gx[:, :, ks[0]], cmap=cmap, vmin=cmin, vmax=cmax)
# #     ax.set_title(f'Slice {ks[0]}')
# #     fig.colorbar(im, ax=ax, label=clab)
# #
# #     ax = fig.add_subplot(132)
# #     im = ax.imshow(gx[:, ks[1], :], cmap=cmap, vmin=cmin, vmax=cmax)
# #     ax.set_title(f'Slice {ks[1]}')
# #     fig.colorbar(im, ax=ax, label=clab)
# #
# #     ax = fig.add_subplot(133)
# #     im = ax.imshow(gx[ks[2], :, :], cmap=cmap, vmin=cmin, vmax=cmax)
# #     ax.set_title(f'Slice {ks[2]}')
# #     fig.colorbar(im, ax=ax, label=clab)
# #
# #     if png:
# #         plt.savefig(f'{png}.png')
# #
# #     plt.show()
# #
# #     if fp is not None:
# #         fig = plt.figure(figsize=(15, 10))
# #         ax = fig.add_subplot(131)
# #         im = ax.imshow(fp[:, :, ks[0]], cmap=cmap, vmin=cmin, vmax=cmax)
# #         ax.set_title(f'Slice {ks[0]}')
# #         fig.colorbar(im, ax=ax, label=clab)
# #
# #         ax = fig.add_subplot(132)
# #         im = ax.imshow(fp[:, ks[1], :], cmap=cmap, vmin=cmin, vmax=cmax)
# #         ax.set_title(f'Slice {ks[1]}')
# #         fig.colorbar(im, ax=ax, label=clab)
# #
# #         ax = fig.add_subplot(133)
# #         im = ax.imshow(fp[ks[2], :, :], cmap=cmap, vmin=cmin, vmax=cmax)
# #         ax.set_title(f'Slice {ks[2]}')
# #         fig.colorbar(im, ax=ax, label=clab)
# #
# #         if png:
# #             plt.savefig(f'{png}_fp.png')
# #
# #         plt.show()
# #
# # if __name__ == '__main__':
# #     main(sys.argv)


# # Java库转python
# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.mplot3d import Axes3D
# import scipy.ndimage as ndimage
# import pickle

# #############################################################################
# # Internal constants

# _datdir = "../data/"
# _pngdir = "../png1/"
# #############################################################################
# def main(args):
#     if args[1] == "jie":
#         goJie()
#     elif args[1] == "hongliu":
#         goHongliu()
#     elif args[1] == "valid":
#         goValid(args[2])
#     else:
#         print("demo not found")
# def setupForSubset(name):
#     global pngDir
#     global seismicDir
#     global s1, s2, s3
#     global n1, n2, n3
#     if name == "jie":
#         print("setupForSubset: jie")
#         pngDir = os.path.join(_pngdir, "jie/")
#         seismicDir = os.path.join(_datdir, "jie/")
#         n1, n2, n3 =128, 256,512
#         d1, d2, d3 = 1, 1, 1  # (s,km/s)
#         f1, f2, f3 = 0, 0, 0
#         s1, s2, s3 = np.linspace(f1, f1 + (n1 - 1) * d1, n1), np.linspace(f2, f2 + (n2 - 1) * d2, n2), np.linspace(f3, f3 + (n3 - 1) * d3, n3)
#     elif name == "validation":
#         print("setupForSubset: validation")
#         pngDir = os.path.join(_pngdir, "validation/")
#         seismicDir = os.path.join(_datdir, "validation/")
#         n1, n2, n3 = 256, 256, 256
#         d1, d2, d3 = 1.0, 1.0, 1.0
#         f1, f2, f3 = 0.0, 0.0, 0.0
#         # s1 = np.linspace(0, 255, 256)
#         # s2 = np.linspace(0, 255, 256)
#         # s3 = np.linspace(0, 255, 256)
#         s1, s2, s3 = np.linspace(f1, f1 + (n1 - 1) * d1, n1), np.linspace(f2, f2 + (n2 - 1) * d2, n2), np.linspace(f3, f3 + (n3 - 1) * d3, n3)
#     elif name == "hongliu":
#         print("setupForSubset: hongliu")
#         pngDir = os.path.join(_pngdir, "hongliu/")
#         seismicDir = os.path.join(_datdir, "hongliu/")
#         n1, n2, n3 = 256, 256, 256
#         d1, d2, d3 = 1.0, 1.0, 1.0
#         f1, f2, f3 = 0.0, 0.0, 0.0
#         s1, s2, s3 = np.linspace(f1, f1 + (n1 - 1) * d1, n1), np.linspace(f2, f2 + (n2 - 1) * d2, n2), np.linspace(f3, f3 + (n3 - 1) * d3, n3)
#     else:
#         print("unrecognized subset:", name)
#         sys.exit(1)

# def getSamplings():
#     return s1, s2, s3

# def getSeismicDir():
#     return seismicDir

# def getPngDir():
#     return pngDir

# #############################################################################
# # read/write images

# def readImageChannels(basename):
#     fileName = os.path.join(seismicDir, basename + ".jpg")
#     image = plt.imread(fileName)
#     return image

# def readColorImage(basename):
#     return readImageChannels(basename)

# def readImage2D(n1, n2, basename):
#     fileName = os.path.join(seismicDir, basename + ".dat")
#     image = np.fromfile(fileName, dtype=np.float32).reshape(n2, n1)
#     return image

# def readImage(basename):
#     fileName = os.path.join(seismicDir, basename)
#     # fileName = os.path.join(seismicDir, basename )
#     image = np.fromfile(fileName, dtype=np.float64).reshape(n3, n2, n1)
#     # image = np.fromfile(fileName, dtype=np.float64).reshape(n1, n2, n3)
#     image=np.transpose(image)
#     print('111',image.shape)
#     return image
# # validation
# # def readImage(basename):
# #     fileName = os.path.join(seismicDir, basename + ".dat")
# #     # fileName = os.path.join(seismicDir, basename )
# #     image = np.fromfile(fileName, dtype=np.float32).reshape(n1, n2, n3)
# #     image=np.transpose(image)
# #     print('111',image.shape)
# #     return image
# def readImage1(basename):
#     # fileName = os.path.join(seismicDir, basename + ".dat")
#     fileName = os.path.join(seismicDir, basename )
#     # image = np.fromfile(fileName, dtype=np.float32).reshape(n1, n2, n3)
#     image = np.load(fileName,allow_pickle=True)
#     # image = np.reshape(image, (n3, n2, n1))  # 修改为PyTorch格式 = gx['arr_0']
#     image = np.reshape(image, (n1, n2, n3))  # 修改为PyTorch格式 = gx['arr_0']
#     print('222',image.shape)
#     return image

# def readImage3DB(basename):
#     return readImage(basename)

# def readImageL(basename):
#     return readImage(basename)

# def readImage2DL(basename):
#     return readImage2D(n1, n2, basename)

# def readImage1D(basename):
#     fileName = os.path.join(seismicDir, basename + ".dat")
#     image = np.fromfile(fileName, dtype=np.float32).reshape(n1)
#     return image

# def readImage1L(basename):
#     return readImage1D(basename)

# def writeImage(basename, image):
#     fileName = os.path.join(seismicDir, basename + ".dat")
#     image.tofile(fileName)
#     return image

# def writeImagex(fname, image):
#     fileName = fname + ".dat"
#     image.tofile(fileName)
#     return image

# def writeImageL(basename, image):
#     fileName = os.path.join(seismicDir, basename + ".dat")
#     print(fileName)
#     image.tofile(fileName)
#     return image


# #############################################################################
# # read/write fault skins

# def skinName(basename, index):
#     return basename + ("%05i" % index)

# def skinIndex(basename, fileName):
#     assert fileName.startswith(basename)
#     i = len(basename)
#     return int(fileName[i:i+5])

# def listAllSkinFiles(basename):
#     fileNames = [f for f in os.listdir(seismicDir) if f.startswith(basename)]
#     fileNames.sort()
#     return fileNames

# def removeAllSkinFiles(basename):
#     fileNames = listAllSkinFiles(basename)
#     for fileName in fileNames:
#         os.remove(os.path.join(seismicDir, fileName))

# def readSkin(basename, index):
#     with open(os.path.join(seismicDir, skinName(basename, index) + ".dat"), 'rb') as f:
#         skin = pickle.load(f)
#     return skin

# def getSkinFileNames(basename):
#     return listAllSkinFiles(basename)

# def readSkins(basename):
#     fileNames = listAllSkinFiles(basename)
#     skins = [readSkin(basename, skinIndex(basename, fileName)) for fileName in fileNames]
#     return skins

# def writeSkin(basename, index, skin):
#     with open(os.path.join(seismicDir, skinName(basename, index) + ".dat"), 'wb') as f:
#         pickle.dump(skin, f)

# def writeSkins(basename, skins):
#     for index, skin in enumerate(skins):
#         writeSkin(basename, index, skin)

# def readObject(name):
#     with open(os.path.join(seismicDir, name + ".dat"), 'rb') as f:
#         obj = pickle.load(f)
#     return obj

# def writeObject(name, obj):
#     with open(os.path.join(seismicDir, name + ".dat"), 'wb') as f:
#         pickle.dump(obj, f)





# #############################################################################
# # Additional functions

# def plot3(s1, s2, s3, f, g=None, cmin=-2, cmax=2, zs=0.5, sc=1.4, ks=[175, 330, 377], ae=[45, 35], vt=[-0.1, -0.06, 0.0], cmap=None, clab=None, cint=None, surf=None, png=None):
#     n3, n2, n1 = 129,257,257
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     x, y, z = np.meshgrid(np.arange(n2), np.arange(n3), np.arange(n1))
#     ax.voxels(x, y, z, f > 0, facecolors=plt.cm.jet(f / f.max()))
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     if clab:
#         mappable = plt.cm.ScalarMappable(cmap=cmap)
#         mappable.set_array(f)
#         fig.colorbar(mappable, ax=ax, label=clab)
#     plt.show()


# def plot2(s1, s2, x, slice_idx, cmap='jet', clab="Amplitude", cmin=None, cmax=None, title=None, bottom_title=None, png=None, save_dir="."):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     mid_slice = x[slice_idx,:, :]
#     mid_slice = mid_slice[0:256, 0:256]

#     im = ax.imshow(mid_slice, cmap=cmap, vmin=cmin, vmax=cmax, aspect='auto')

#     # Show x-axis ticks and labels at the top
#     ax.xaxis.set_ticks_position('top')
#     ax.xaxis.set_label_position('top')
#     ax.set_xlabel("crossLine", labelpad=10, fontsize=12)
#     ax.set_ylabel("xLine", labelpad=10, fontsize=12)

#     if title:
#         ax.set_xlabel(title, fontsize=15, labelpad=20)
#         ax.xaxis.set_label_position('bottom')
#         ax.xaxis.set_ticks_position('none')  # Don't show bottom ticks

#     if bottom_title:
#         fig.text(0.45, 0.05, bottom_title, ha='center', fontsize=15)

#     ax.tick_params(axis='x', which='both', top=True, labeltop=True, bottom=False, labelbottom=False)
#     ax.tick_params(axis='both', labelsize=10)

#     if clab:
#         cbar = plt.colorbar(im, ax=ax)
#         cbar.set_label(clab, fontsize=12)

#     # if png:
#     #     if not os.path.exists(save_dir):
#     #         os.makedirs(save_dir)
#     #     plt.savefig(os.path.join(save_dir, f"{png}-slice-t-{slice_idx}.png"))
#     plt.show()
# def plot21(s1, s2, x, slice_idx, cmap='jet', clab="Amplitude", cmin=None, cmax=None, title=None, bottom_title=None,
#            png=None, save_dir="."):
#     fig, ax = plt.subplots(figsize=(12, 6))  # Adjusted figure size
#     mid_slice = x[slice_idx, :, :]
#     # mid_slice = mid_slice[0:252, 0:240]
#     mid_slice = mid_slice[0:256, 0:512]

#     im = ax.imshow(mid_slice, cmap=cmap, vmin=cmin, vmax=cmax, aspect='auto')

#     # Show x-axis ticks and labels at the top
#     ax.xaxis.set_ticks_position('top')
#     ax.xaxis.set_label_position('top')
#     ax.set_xlabel("iLine", labelpad=5, fontsize=12)
#     ax.set_ylabel("xLine", labelpad=10, fontsize=12)

#     if title:
#         ax.set_title(title, fontsize=15, pad=20)

#     if bottom_title:
#         fig.text(0.45, 0.02, bottom_title, ha='center', fontsize=15)

#     ax.tick_params(axis='x', which='both', top=True, labeltop=True, bottom=False, labelbottom=False)
#     ax.tick_params(axis='both', labelsize=10)
#     # ax.set_xticks(np.arange(0, 252, 50))
#     # ax.set_yticks(np.arange(0, 240, 50))

#     if clab:
#         # Create a smaller colorbar
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="2%", pad=0.10)
#         cbar = plt.colorbar(im, cax=cax)
#         cbar.set_label(clab, fontsize=12)
#         cbar.ax.tick_params(labelsize=8)  # Adjust tick size of colorbar

#     if png:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         # np.save(os.path.join(save_dir, f"{png}-slice-t1-{slice_idx}.npy"), mid_slice)
#         plt.savefig(os.path.join(save_dir, f"{png}-slice-t1-{slice_idx}.png"))
#     plt.show()
# def goValid(fname):
#     setupForSubset("validation")
#     global pngDir
#     pngDir = getPngDir()
#     s1, s2, s3 = getSamplings()
#     gx = readImage("nx/" + fname)
#     lx = readImage("lx/" + fname)
#     fx = readImage("px/" + fname)
#     print(gx.shape)
#     save_dir = './png'
#     slice_indices = range(0, gx.shape[0], 5)

#     for slice_idx in slice_indices:
#         plot2(s1, s2, gx, slice_idx, cmap='gray', clab=" ", title=' ', bottom_title=f't={slice_idx}', png=fname+'-gx', save_dir=save_dir)
#         plot2(s1, s2, lx, slice_idx, cmap='gray', clab=" ", title=' ', bottom_title=f't={slice_idx}', png=fname+'-lx', save_dir=save_dir)
#         plot2(s1, s2, fx, slice_idx, cmap='gray', clab=" ", title=' ', bottom_title=f't={slice_idx}', png=fname+'-fp', save_dir=save_dir)

# def goJie():
#     gxfile = r"D:\anaconda3\envs\KarstSeg3D-master\data\MIG_ALLAGC-1.npy"  # input seismic image
#     fpfile = r"D:\anaconda3\envs\KarstSeg3D-master\data\MIG_ALLAGC-1.dat"  # karst probability by cnn
#     setupForSubset("jie")
#     global pngDir
#     pngDir = getPngDir()
#     s1, s2, s3 = getSamplings()
#     gx = readImage1(gxfile)
#     fp = readImage(fpfile)
#     ks = [168, 345, 752]
#     vt = [-0.25, -0.48, 0.0]
#     ae = [-45, 50]
#     save_dir = './png1'
#     slice_indices = range(0, gx.shape[0], 4)
#     # plot3(s1, s2, s3, gx, cmin=-2, cmax=2, ks=ks, ae=ae, vt=vt, clab="Amplitude", png="gx")
#     # plot3(s1, s2, s3, gx, fp, cmin=0.2, cmax=1, ks=ks, ae=ae, vt=vt, cmap=plt.cm.jet, clab="Karst probability", png="fp")
#     for slice_idx in slice_indices:
#         plot21(s1, s2, gx,slice_idx,  cmap='jet',clab=' ', cmin=None, cmax=None, title=None,bottom_title=None,
#               png='MIG_ALLAGC-UNet-gx',save_dir=save_dir)
#         plot21(s1, s2, fp,slice_idx, cmap='jet',clab=' ', cmin=None, cmax=None, title=None,bottom_title=None,
#               png='MIG_ALLAGCUNet-fp',save_dir=save_dir)
# def goHongliu():
#     gxfile = "gg"  # input seismic image
#     fpfile = "fp"  # karst probability by cnn
#     setupForSubset("hongliu")
#     global pngDir
#     pngDir = getPngDir()
#     s1, s2, s3 = getSamplings()
#     gx = readImage(gxfile)
#     fp = readImage(fpfile)
#     ks = [174, 195, 60]
#     ae = [135, 35]
#     vt = [-0.0, -0.0, 0.0]
#     plot3(s1, s2, s3, gx, cmin=-2, cmax=2, ks=ks, ae=ae, vt=vt, clab="Amplitude", png="gx")
#     plot3(s1, s2, s3, gx, fp, cmin=0.2, cmax=1, ks=ks, ae=ae, vt=vt, cmap=plt.cm.jet, clab="Karst probability", png="fp")

# #############################################################################
# # Run the function main on the main thread
# if __name__ == "__main__":
#     main(sys.argv)


# #############################################################################
# """
# Demo of demostrating fault results
# Author: Xinming Wu, USTC
# Version: 2020.03.26
# """
# from keras.src.ops import copy
#
# from datUtils import *
# #############################################################################
#
# plotOnly = True
# plotOnly = False
#
# def main(args):
#   if args[1]=="jie":
#     goJie()
#   elif args[1]=="hongliu":
#     goHongliu()
#   elif args[1]=="valid":
#     goValid(args[2])
#   else:
#     print("demo not found")
#
# def goValid(fname):
#   setupForSubset("validation")
#   global pngDir
#   pngDir = getPngDir()
#   s1,s2,s3 = getSamplings()
#   gx = readImage("nx/"+fname)
#   fp = readImage("px/"+fname)
#   ks = [220,212,50]
#   vt=[-0.0,-0.0,0.0]
#   ae=[125,20]
#   plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=ae,vt=vt,clab="Amplitude",png="gx")
#   plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=ae,vt=vt,cmap=jetRamp(1.0),
#           clab="Karst probability",png="fp")
#
# def goJie():
#   gxfile = "gg" # input seismic image
#   fpfile = "fp" # karst probability by cnn
#   setupForSubset("jie")
#   global pngDir
#   pngDir = getPngDir()
#   s1,s2,s3 = getSamplings()
#   gx = readImage(gxfile)
#   fp = readImage(fpfile)
#   ks = [168,345,752]
#   vt=[-0.25,-0.48,0.0]
#   ae=[-45,50]
#   plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=ae,vt=vt,clab="Amplitude",png="gx")
#   plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=ae,vt=vt,cmap=jetRamp(1.0),
#           clab="Karst probability",png="fp")
# def goHongliu():
#   gxfile = "gg" # input seismic image
#   fpfile = "fp" # karst probability by cnn
#   setupForSubset("hongliu")
#   global pngDir
#   pngDir = getPngDir()
#   s1,s2,s3 = getSamplings()
#   gx = readImage(gxfile)
#   fp = readImage(fpfile)
#   ks = [174,195,60]
#   ae = [135,35]
#   vt = [-0.0,-0.0,0.0]
#   plot3(s1,s2,s3,gx,cmin=-2,cmax=2,ks=ks,ae=ae,vt=vt,clab="Amplitude",png="gx")
#   plot3(s1,s2,s3,gx,fp,cmin=0.2,cmax=1,ks=ks,ae=ae,vt=vt,cmap=jetRamp(1.0),
#           clab="Karst probability",png="fp")
#
# def gain(x):
#   g = mul(x,x)
#   ref = RecursiveExponentialFilter(20.0)
#   ref.apply1(g,g)
#   div(x,sqrt(g),x)
#   return x
#
# def checkNaN(gx):
#   n3 = len(gx)
#   n2 = len(gx[0])
#   n1 = len(gx[0][0])
#   for i3 in range(n3):
#     for i2 in range(n2):
#       for i1 in range(n1):
#         if(gx[i3][i2][i1]!=gx[i3][i2][i1]):
#           gx[i3][i2][i1] = 0
#   return gx
#
# def smooth(sig,u):
#   v = copy(u)
#   rgf = RecursiveGaussianFilterP(sig)
#   rgf.apply0(u,v)
#   return v
#
# def smooth2(sig1,sig2,u):
#   v = copy(u)
#   rgf1 = RecursiveGaussianFilterP(sig1)
#   rgf2 = RecursiveGaussianFilterP(sig2)
#   rgf1.apply0X(u,v)
#   rgf2.applyX0(v,v)
#   return v
#
# def normalize(e):
#   emin = min(e)
#   emax = max(e)
#   return mul(sub(e,emin),1.0/(emax-emin))
#
# def slice12(k3,f):
#   n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
#   s = zerofloat(n1,n2)
#   SimpleFloat3(f).get12(n1,n2,0,0,k3,s)
#   return s
#
# def slice13(k2,f):
#   n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
#   s = zerofloat(n1,n3)
#   SimpleFloat3(f).get13(n1,n3,0,k2,0,s)
#   return s
#
# def slice23(k1,f):
#   n1,n2,n3 = len(f[0][0]),len(f[0]),len(f)
#   s = zerofloat(n2,n3)
#   SimpleFloat3(f).get23(n2,n3,k1,0,0,s)
#   return s
#
# #############################################################################
# # graphics
#
# def jetFill(alpha):
#   return ColorMap.setAlpha(ColorMap.JET,alpha)
# def jetFillExceptMin(alpha):
#   a = fillfloat(alpha,256)
#   a[0] = 0.0
#   return ColorMap.setAlpha(ColorMap.JET,a)
# def jetRamp(alpha):
#   return ColorMap.setAlpha(ColorMap.JET,rampfloat(0.0,alpha/256,256))
# def bwrFill(alpha):
#   return ColorMap.setAlpha(ColorMap.BLUE_WHITE_RED,alpha)
# def bwrNotch(alpha):
#   a = zerofloat(256)
#   for i in range(len(a)):
#     if i<128:
#       a[i] = alpha*(128.0-i)/128.0
#     else:
#       a[i] = alpha*(i-127.0)/128.0
#   return ColorMap.setAlpha(ColorMap.BLUE_WHITE_RED,a)
# def hueFill(alpha):
#   return ColorMap.getHue(0.0,1.0,alpha)
# def hueFillExceptMin(alpha):
#   a = fillfloat(alpha,256)
#   a[0] = 0.0
#   return ColorMap.setAlpha(ColorMap.getHue(0.0,1.0),a)
#
# def addColorBar(frame,clab=None,cint=None):
#   cbar = ColorBar(clab)
#   if cint:
#     cbar.setInterval(cint)
#   cbar.setFont(Font("Arial",Font.PLAIN,32)) # size by experimenting
#   cbar.setWidthMinimum
#   cbar.setBackground(Color.WHITE)
#   frame.add(cbar,BorderLayout.EAST)
#   return cbar
#
# def convertDips(ft):
#   return FaultScanner.convertDips(0.2,ft) # 5:1 vertical exaggeration
#
# def plot2(s1,s2,x,u=None,g=None,x1=None,c=None,
#         cmap=ColorMap.GRAY,clab="Amplitude",
#         cmin=-2,cmax=2,title=None,png=None):
#   sp = SimplePlot(SimplePlot.Origin.UPPER_LEFT)
#   if title:
#     sp.setTitle(title)
#   n1,n2=s1.count,s2.count
#   sp.addColorBar(clab)
#   #sp.setSize(955,400)
#   sp.setSize(755,500)
#   sp.setHLabel("Inline (sample)")
#   sp.setVLabel("Depth (sample)")
#   sp.plotPanel.setColorBarWidthMinimum(60)
#   sp.setVLimits(0,n1-1)
#   sp.setHLimits(0,n2-1)
#   sp.setFontSize(16)
#   pv = sp.addPixels(s1,s2,x)
#   pv.setColorModel(cmap)
#   pv.setInterpolation(PixelsView.Interpolation.LINEAR)
#   if cmin<cmax:
#     pv.setClips(cmin,cmax)
#   if u:
#     cv = sp.addContours(s1,s2,u)
#     cv.setContours(80)
#     cv.setLineColor(Color.YELLOW)
#   if g:
#     pv = sp.addPixels(s1,s2,g)
#     pv.setInterpolation(PixelsView.Interpolation.NEAREST)
#     pv.setColorModel(ColorMap.getJet(0.8))
#     pv.setClips(0.1,s1.count)
#   if x1:
#     x1k = zerofloat(n2)
#     x2  = zerofloat(n2)
#     x1s  = zerofloat(n1)
#     for i1 in range(n1):
#       x1s[i1] = i1
#     cp = ColorMap(0,n1,ColorMap.JET)
#     rgb = cp.getRgbFloats(x1s)
#     ref = RecursiveExponentialFilter(1)
#     for k in range(20,n1-20,15):
#       for i2 in range(n2):
#         x2[i2] = i2
#         x1k[i2] = x1[i2][k]
#       ref.apply(x1k,x1k)
#       pv = sp.addPoints(x1k,x2)
#       pv.setLineWidth(2.5)
#       r,g,b=rgb[k*3],rgb[k*3+1],rgb[k*3+2]
#       pv.setLineColor(Color(r,g,b))
#   if pngDir and png:
#     sp.paintToPng(700,3.333,pngDir+png+".png")
#
# def plot3(s1,s2,s3,f,g=None,cmin=-2,cmax=2,zs=0.5,sc=1.4,
#         ks=[175,330,377],ae=[45,35],vt=[-0.1,-0.06,0.0],
#         cmap=None,clab=None,cint=None,surf=None,png=None):
#   n3 = len(f)
#   n2 = len(f[0])
#   n1 = len(f[0][0])
#   d1,d2,d3 = s1.delta,s2.delta,s3.delta
#   f1,f2,f3 = s1.first,s2.first,s3.first
#   l1,l2,l3 = s1.last,s2.last,s3.last
#   sf = SimpleFrame(AxesOrientation.XRIGHT_YOUT_ZDOWN)
#   cbar = None
#   if g==None:
#     ipg = sf.addImagePanels(s1,s2,s3,f)
#     if cmap!=None:
#       ipg.setColorModel(cmap)
#     if cmin!=None and cmax!=None:
#       ipg.setClips(cmin,cmax)
#     else:
#       ipg.setClips(-2.0,2.0)
#     if clab:
#       cbar = addColorBar(sf,clab,cint)
#       ipg.addColorMapListener(cbar)
#   else:
#     ipg = ImagePanelGroup2(s1,s2,s3,f,g)
#     ipg.setClips1(-2,2)
#     if cmin!=None and cmax!=None:
#       ipg.setClips2(cmin,cmax)
#     if cmap==None:
#       cmap = jetFill(0.8)
#     ipg.setColorModel2(cmap)
#     if clab:
#       cbar = addColorBar(sf,clab,cint)
#       ipg.addColorMap2Listener(cbar)
#     sf.world.addChild(ipg)
#   if cbar:
#     cbar.setWidthMinimum(120)
#   if surf:
#     tg = TriangleGroup(True,surf)
#     sf.world.addChild(tg)
#   ipg.setSlices(ks[0],ks[1],ks[2])
#   '''
#   if cbar:
#     sf.setSize(987,720)
#   else:
#     sf.setSize(850,720)
#   '''
#   if cbar:
#     sf.setSize(887,750)
#   else:
#     sf.setSize(750,750)
#
#   vc = sf.getViewCanvas()
#   vc.setBackground(Color.WHITE)
#   radius = 0.5*sqrt(n1*n1+n2*n2+n3*n3)
#   ov = sf.getOrbitView()
#   zscale = zs*max(n2*d2,n3*d3)/(n1*d1)
#   #ov.setAxesScale(1.0,1.0,zscale)
#   ov.setScale(sc)
#   ov.setWorldSphere(BoundingSphere(0.5*n1,0.5*n2,0.5*n3,radius))
#   #ov.setWorldSphere(BoundingSphere(BoundingBox(f3,f2,f1,l3,l2,l1)))
#   ov.setTranslate(Vector3(vt[0],vt[1],vt[2]))
#   ov.setAzimuthAndElevation(ae[0],ae[1])
#   sf.setVisible(True)
#   if png and pngDir:
#     sf.paintToFile(pngDir+png+".png")
#     if cbar:
#       cbar.paintToPng(720,1,pngDir+png+"cbar.png")
# #############################################################################
# # Run the function main on the Swing thread
# import sys
# class _RunMain(Runnable):
#   def __init__(self,main):
#     self.main = main
#   def run(self):
#     self.main(sys.argv)
# def run(main):
#   SwingUtilities.invokeLater(_RunMain(main))
# run(main)
#


#############################################################################
"""
Demo of demonstrating fault results
Author: Xinming Wu, USTC
Version: 2020.03.26
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
#
# from datUtils import *
# #############################################################################
#
# plotOnly = True
# plotOnly = False
#
# def main(args):
#     if args[1] == "jie":
#         go_jie()
#     elif args[1] == "hongliu":
#         go_hongliu()
#     elif args[1] == "valid":
#         go_valid(args[2])
#     else:
#         print("demo not found")
#
# def load_model(mk):
#     global model
#     model = torch.load(f'check/checkpoint.{mk}.pth')
#     model.eval()
#
# def read_image(path):
#     image = np.fromfile(path, dtype=np.float32)
#     image = image.reshape((320, 1024, 1024))
#     return image
#
# def go_valid(fname):
#     setup_for_subset("validation")
#     global png_dir
#     png_dir = get_png_dir()
#     s1, s2, s3 = get_samplings()
#     gx = read_image(f"nx/{fname}.dat")
#     fp = read_image(f"px/{fname}.dat")
#     ks = [220, 212, 50]
#     vt = [-0.0, -0.0, 0.0]
#     ae = [125, 20]
#     plot3(s1, s2, s3, gx, cmin=-2, cmax=2, ks=ks, ae=ae, vt=vt, clab="Amplitude", png="gx")
#     plot3(s1, s2, s3, gx, fp, cmin=0.2, cmax=1, ks=ks, ae=ae, vt=vt, cmap='jet',
#           clab="Karst probability", png="fp")
#
# def go_jie():
#     gxfile = "gg"  # input seismic image
#     fpfile = "fp"  # karst probability by cnn
#     setup_for_subset("jie")
#     global png_dir
#     png_dir = get_png_dir()
#     s1, s2, s3 = get_samplings()
#     gx = read_image(gxfile)
#     fp = read_image(fpfile)
#     ks = [168, 345, 752]
#     vt = [-0.25, -0.48, 0.0]
#     ae = [-45, 50]
#     plot3(s1, s2, s3, gx, cmin=-2, cmax=2, ks=ks, ae=ae, vt=vt, clab="Amplitude", png="gx")
#     plot3(s1, s2, s3, gx, fp, cmin=0.2, cmax=1, ks=ks, ae=ae, vt=vt, cmap='jet',
#           clab="Karst probability", png="fp")
#
# def go_hongliu():
#     gxfile = "gg"  # input seismic image
#     fpfile = "fp"  # karst probability by cnn
#     setup_for_subset("hongliu")
#     global png_dir
#     png_dir = get_png_dir()
#     s1, s2, s3 = get_samplings()
#     gx = read_image(gxfile)
#     fp = read_image(fpfile)
#     ks = [174, 195, 60]
#     ae = [135, 35]
#     vt = [-0.0, -0.0, 0.0]
#     plot3(s1, s2, s3, gx, cmin=-2, cmax=2, ks=ks, ae=ae, vt=vt, clab="Amplitude", png="gx")
#     plot3(s1, s2, s3, gx, fp, cmin=0.2, cmax=1, ks=ks, ae=ae, vt=vt, cmap='jet',
#           clab="Karst probability", png="fp")
#
# def plot3(s1, s2, s3, gx, fp=None, cmin=-2, cmax=2, ks=[175, 330, 377], ae=[45, 35], vt=[-0.1, -0.06, 0.0], cmap='gray', clab=None, png=None):
#     fig = plt.figure(figsize=(15, 10))
#     ax = fig.add_subplot(131)
#     im = ax.imshow(gx[:, :, ks[0]], cmap=cmap, vmin=cmin, vmax=cmax)
#     ax.set_title(f'Slice {ks[0]}')
#     fig.colorbar(im, ax=ax, label=clab)
#
#     ax = fig.add_subplot(132)
#     im = ax.imshow(gx[:, ks[1], :], cmap=cmap, vmin=cmin, vmax=cmax)
#     ax.set_title(f'Slice {ks[1]}')
#     fig.colorbar(im, ax=ax, label=clab)
#
#     ax = fig.add_subplot(133)
#     im = ax.imshow(gx[ks[2], :, :], cmap=cmap, vmin=cmin, vmax=cmax)
#     ax.set_title(f'Slice {ks[2]}')
#     fig.colorbar(im, ax=ax, label=clab)
#
#     if png:
#         plt.savefig(f'{png}.png')
#
#     plt.show()
#
#     if fp is not None:
#         fig = plt.figure(figsize=(15, 10))
#         ax = fig.add_subplot(131)
#         im = ax.imshow(fp[:, :, ks[0]], cmap=cmap, vmin=cmin, vmax=cmax)
#         ax.set_title(f'Slice {ks[0]}')
#         fig.colorbar(im, ax=ax, label=clab)
#
#         ax = fig.add_subplot(132)
#         im = ax.imshow(fp[:, ks[1], :], cmap=cmap, vmin=cmin, vmax=cmax)
#         ax.set_title(f'Slice {ks[1]}')
#         fig.colorbar(im, ax=ax, label=clab)
#
#         ax = fig.add_subplot(133)
#         im = ax.imshow(fp[ks[2], :, :], cmap=cmap, vmin=cmin, vmax=cmax)
#         ax.set_title(f'Slice {ks[2]}')
#         fig.colorbar(im, ax=ax, label=clab)
#
#         if png:
#             plt.savefig(f'{png}_fp.png')
#
#         plt.show()
#
# if __name__ == '__main__':
#     main(sys.argv)


# Java库转python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as ndimage
import pickle

#############################################################################
# Internal constants

_datdir = "../data/"
_pngdir = "../png1/"
#############################################################################
def main(args):
    if args[1] == "jie":
        goJie()
    elif args[1] == "hongliu":
        goHongliu()
    elif args[1] == "valid":
        goValid(args[2])
    else:
        print("demo not found")
def setupForSubset(name):
    global pngDir
    global seismicDir
    global s1, s2, s3
    global n1, n2, n3
    if name == "jie":
        print("setupForSubset: jie")
        pngDir = os.path.join(_pngdir, "jie/")
        seismicDir = os.path.join(_datdir, "jie/")
        n1, n2, n3 =128, 256,512
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
        # s1 = np.linspace(0, 255, 256)
        # s2 = np.linspace(0, 255, 256)
        # s3 = np.linspace(0, 255, 256)
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
    fileName = os.path.join(seismicDir, basename + ".jpg")
    image = plt.imread(fileName)
    return image

def readColorImage(basename):
    return readImageChannels(basename)

def readImage2D(n1, n2, basename):
    fileName = os.path.join(seismicDir, basename + ".dat")
    image = np.fromfile(fileName, dtype=np.float32).reshape(n2, n1)
    return image

def readImage(basename):
    fileName = os.path.join(seismicDir, basename)
    # fileName = os.path.join(seismicDir, basename )
    image = np.fromfile(fileName, dtype=np.float64).reshape(n3, n2, n1)
    # image = np.fromfile(fileName, dtype=np.float64).reshape(n1, n2, n3)
    image=np.transpose(image)
    print('111',image.shape)
    return image
# # validation
# def readImage(basename):
#     fileName = os.path.join(seismicDir, basename + ".dat")
#     # fileName = os.path.join(seismicDir, basename )
#     image = np.fromfile(fileName, dtype=np.float32).reshape(n1, n2, n3)
#     image=np.transpose(image)
#     print('111',image.shape)
#     return image
def readImage1(basename):
    # fileName = os.path.join(seismicDir, basename + ".dat")
    fileName = os.path.join(seismicDir, basename )
    # image = np.fromfile(fileName, dtype=np.float32).reshape(n1, n2, n3)
    image = np.load(fileName,allow_pickle=True)
    # image = np.reshape(image, (n3, n2, n1))  # 修改为PyTorch格式 = gx['arr_0']
    image = np.reshape(image, (n1, n2, n3))  # 修改为PyTorch格式 = gx['arr_0']
    print('222',image.shape)
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
    fileNames = [f for f in os.listdir(seismicDir) if f.startswith(basename)]
    fileNames.sort()
    return fileNames

def removeAllSkinFiles(basename):
    fileNames = listAllSkinFiles(basename)
    for fileName in fileNames:
        os.remove(os.path.join(seismicDir, fileName))

def readSkin(basename, index):
    with open(os.path.join(seismicDir, skinName(basename, index) + ".dat"), 'rb') as f:
        skin = pickle.load(f)
    return skin

def getSkinFileNames(basename):
    return listAllSkinFiles(basename)

def readSkins(basename):
    fileNames = listAllSkinFiles(basename)
    skins = [readSkin(basename, skinIndex(basename, fileName)) for fileName in fileNames]
    return skins

def writeSkin(basename, index, skin):
    with open(os.path.join(seismicDir, skinName(basename, index) + ".dat"), 'wb') as f:
        pickle.dump(skin, f)

def writeSkins(basename, skins):
    for index, skin in enumerate(skins):
        writeSkin(basename, index, skin)

def readObject(name):
    with open(os.path.join(seismicDir, name + ".dat"), 'rb') as f:
        obj = pickle.load(f)
    return obj

def writeObject(name, obj):
    with open(os.path.join(seismicDir, name + ".dat"), 'wb') as f:
        pickle.dump(obj, f)





#############################################################################
# Additional functions

def plot3(s1, s2, s3, f, g=None, cmin=-2, cmax=2, zs=0.5, sc=1.4, ks=[175, 330, 377], ae=[45, 35], vt=[-0.1, -0.06, 0.0], cmap=None, clab=None, cint=None, surf=None, png=None):
    n3, n2, n1 = 129,257,257
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.meshgrid(np.arange(n2), np.arange(n3), np.arange(n1))
    ax.voxels(x, y, z, f > 0, facecolors=plt.cm.jet(f / f.max()))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if clab:
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_array(f)
        fig.colorbar(mappable, ax=ax, label=clab)
    plt.show()

def plot2(s1, s2, x, slice_idx, cmap='jet', clab="Amplitude", cmin=None, cmax=None, title=None, bottom_title=None, png=None, save_dir="."):
    fig, ax = plt.subplots(figsize=(8, 8))
    mid_slice = x[slice_idx,:, :]
    mid_slice = mid_slice[0:252, 0:240]

    im = ax.imshow(mid_slice, cmap=cmap, vmin=cmin, vmax=cmax, aspect='auto')

    # Show x-axis ticks and labels at the top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("crossLine", labelpad=10, fontsize=12)
    ax.set_ylabel("xLine", labelpad=10, fontsize=12)

    if title:
        ax.set_xlabel(title, fontsize=15, labelpad=20)
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.set_ticks_position('none')  # Don't show bottom ticks

    if bottom_title:
        fig.text(0.45, 0.05, bottom_title, ha='center', fontsize=15)

    ax.tick_params(axis='x', which='both', top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis='both', labelsize=10)

    if clab:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(clab, fontsize=12)

    if png:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"08{png}-slice-t-{slice_idx}.png"))

    plt.show()
def plot21(s1, s2, x, slice_idx, cmap='jet', clab="Amplitude", cmin=None, cmax=None, title=None, bottom_title=None,
           png=None, save_dir="."):
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjusted figure size
    mid_slice = x[slice_idx, :, :]
    mid_slice = mid_slice[0:256, 0:512]
    im = ax.imshow(mid_slice, cmap=cmap, vmin=cmin, vmax=cmax, aspect='auto')

    # Show x-axis ticks and labels at the bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.set_xlabel(" ", labelpad=5, fontsize=12)
    ax.set_ylabel(" ", labelpad=10, fontsize=12)

    if title:
        ax.set_title(title, fontsize=15, pad=20)

    if bottom_title:
        fig.text(0.45, 0.02, bottom_title, ha='center', fontsize=15)

    ax.tick_params(axis='x', which='both', top=True, labeltop=False, bottom=True, labelbottom=True)
    ax.tick_params(axis='both', labelsize=10)

    if clab:
        # Create a smaller colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.10)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(clab, fontsize=12)
        cbar.ax.tick_params(labelsize=8)  # Adjust tick size of colorbar

    if png:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, f"{png}-slice-t1-{slice_idx}.npy"), mid_slice)
        plt.savefig(os.path.join(save_dir, f"{png}-slice-t1-{slice_idx}.png"))

    plt.show()

def goValid(fname):
    setupForSubset("validation")
    global pngDir
    pngDir = getPngDir()
    s1, s2, s3 = getSamplings()
    gx = readImage("nx/" + fname)
    lx = readImage("lx/" + fname)
    fx = readImage("px/" + fname)  # 这里读取的是模型的原始预测概率
    print(gx.shape)
    save_dir = './png'
    slice_indices = range(0, gx.shape[0], 5)

    for slice_idx in slice_indices:
        plot2(s1, s2, gx, slice_idx, cmap='gray', clab=" ", title=' ', bottom_title=f't={slice_idx}', png=fname+'-gx', save_dir=save_dir)
        plot2(s1, s2, lx, slice_idx, cmap='gray', clab=" ", title=' ', bottom_title=f't={slice_idx}', png=fname+'-lx', save_dir=save_dir)
        plot2(s1, s2, fx, slice_idx, cmap='gray', clab=" ", title=' ', bottom_title=f't={slice_idx}', png=fname+'-fp', save_dir=save_dir)


def goJie():
    gxfile = r"D:\anaconda3\envs\KarstSeg3D-master\data\MIG_ALLAGC-1.npy"  # input seismic image
    fpfile = r"D:\anaconda3\envs\KarstSeg3D-master\data\MIG_ALLAGC-1---50.dat"  # karst probability by cnn
    setupForSubset("jie")
    global pngDir
    pngDir = getPngDir()
    s1, s2, s3 = getSamplings()
    gx = readImage1(gxfile)
    fp = readImage(fpfile)
    ks = [168, 345, 752]
    vt = [-0.25, -0.48, 0.0]
    ae = [-45, 50]
    save_dir = '/root/autodl-tmp/png1'
    slice_indices = range(0, gx.shape[0], 4)
    # plot3(s1, s2, s3, gx, cmin=-2, cmax=2, ks=ks, ae=ae, vt=vt, clab="Amplitude", png="gx")
    # plot3(s1, s2, s3, gx, fp, cmin=0.2, cmax=1, ks=ks, ae=ae, vt=vt, cmap=plt.cm.jet, clab="Karst probability", png="fp")
    for slice_idx in slice_indices:
        plot21(s1, s2, gx,slice_idx,  cmap='seismic',clab=' ', cmin=None, cmax=None, title=None,bottom_title=None,
              png='MIG_ALLAGC-UNet--50--gx',save_dir=save_dir)
        plot21(s1, s2, fp,slice_idx, cmap='jet',clab=' ', cmin=None, cmax=None, title=None,bottom_title=None,
              png='MIG_ALLAGC-UNet--50--fp',save_dir=save_dir)
def goHongliu():
    gxfile = "gg"  # input seismic image
    fpfile = "fp"  # karst probability by cnn
    setupForSubset("hongliu")
    global pngDir
    pngDir = getPngDir()
    s1, s2, s3 = getSamplings()
    gx = readImage(gxfile)
    fp = readImage(fpfile)
    ks = [174, 195, 60]
    ae = [135, 35]
    vt = [-0.0, -0.0, 0.0]
    plot3(s1, s2, s3, gx, cmin=-2, cmax=2, ks=ks, ae=ae, vt=vt, clab="Amplitude", png="gx")
    plot3(s1, s2, s3, gx, fp, cmin=0.2, cmax=1, ks=ks, ae=ae, vt=vt, cmap=plt.cm.jet, clab="Karst probability", png="fp")

#############################################################################
# Run the function main on the main thread
if __name__ == "__main__":
    main(sys.argv)
