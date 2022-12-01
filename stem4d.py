'''
*********************************************************************
Ptychography code for WDD reconstruction and aberration measurement.
*********************************************************************
The codes were originally developed by
Katsuaki Nakazawa(ICYS fellow at NIMS, Japan)
e-mail:NAKAZAWA.Katsuaki@nims.go.jp
and debugged and modified by
Kazutaka Mitsuishi (NIMS, Japan)
e-mail:MITSUISHI.Kazutaka@nims.go.jp
based on the following papers and books:

Pennycook, T. J. et al. Efficient phase contrast imaging in STEM using a pixelated detector.
Part 1: experimental demonstration at atomic resolution. Ultramicroscopy 151, 160-167,
doi:10.1016/j.ultramic.2014.09.013 (2015).
Yang, H. et al. Simultaneous atomic-resolution electron ptychography and Z-contrast imaging of
light and heavy elements in complex nanostructures. Nat Commun 7, 12532,
doi:10.1038/ncomms12532 (2016).
Rodenburg, J. & Maiden, A. in Springer handbook of microscopy
(eds P. W. Hawkes & John C. H. Spence) Ch. 17, (Springer Nature, 2019).
Rodenburg, J. M. Advances in Imaging and Electron Physics 87-184 (2008).

*********************************************************************
*Disclaimer*
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*********************************************************************
'''



import glob
import math
import os
import shutil
import numpy as np
import configparser
from PIL import Image
from multiprocessing import Pool
import time
from datetime import datetime

class Image4d:
    def __init__(self, foldername):
        if foldername[-1] != '/':
            foldername += '/'
        self.foldername = foldername

    def convert_image(self, fid):
        frameheader = np.fromfile(fid,dtype=np.uint8,count=64)
        image = np.fromfile(fid, '<H', count=self.ky*self.kx).reshape(self.ky//2, self.kx*2)
        image = self.recompose_image(image)
        return image

    def recompose_image(self, imgin):
        imgout = np.zeros((self.ky, self.kx),dtype=np.int16)
        imgout[0:self.ky//2, 0:self.kx] = imgin[:,0:self.kx]
        imgout[self.ky//2:self.ky, 0:self.kx] = np.rot90(imgin[:,self.kx:self.kx*2], 2)
        return imgout

    def bin_lineimage(self, lineimage):
        lineimage_bin = lineimage.reshape(self.x,
                                          self.ky,
                                          self.kx//self.bin,
                                          self.bin).sum(3)
        return lineimage_bin

    def make_circularfilter(self,
                            cy,
                            cx,
                            i_angle, #detector inner angle
                            o_angle, # detector outer angle
                            azu_angles = np.array([[0, 360]])):
        d_filter = np.zeros((self.ky, self.kx//self.bin))
        a_filter = np.zeros((self.ky, self.kx//self.bin))
        x_array = np.arange((self.kx//self.bin))-(cx)
        y_array = ((cy)-np.arange((self.ky))).reshape(-1,1)
        d_array = np.sqrt(np.square(x_array)
                         +np.square(y_array))*self.pixelsize_k
        if self.kx//self.bin*self.pixelsize_k/2 <= o_angle:
            pass

        a_array = np.arctan2(x_array, y_array)*180 /np.pi + 180
        d_filter[np.where((d_array > i_angle) & (d_array<= o_angle))] = 1
        for angle in range(azu_angles.shape[0]):
            a_filter[np.where((a_array >= azu_angles[angle][0])
                             &(a_array <= azu_angles[angle][1]))] = 1
        c_filter = d_filter*a_filter
        return c_filter

    def make_circularfilter_circle(self,
                            cy,
                            cx,
                            o_angle,
                            ):
        d_filter = np.zeros((self.ky, self.kx//self.bin))
        x_array = np.arange((self.kx//self.bin))-(cx)
        y_array = ((cy)-np.arange((self.ky))).reshape(-1,1)
        d_array = np.sqrt(np.square(x_array)
                         +np.square(y_array))*self.pixelsize_k
        if self.kx//self.bin*self.pixelsize_k/2 <= o_angle:
            print("o_angle is too large.Should be smaller than ", self.kx//self.bin*self.pixelsize_k/2)
        d_filter[np.where((d_array > o_angle) & (d_array<= o_angle+1))] = 1
        c_filter = d_filter
        return c_filter
