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


from multiprocessing import Pool
from multiprocessing import Process

import os
import glob
import math
import stem4d as stem4d
import cv2
import numpy as np
import hyperspy.api as hs
from PIL import Image
from scipy import ndimage
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
import cupy as cp
import sys
import time
from datetime import datetime
import gc
from multiprocessing import Pool
import importlib

importlib.reload(stem4d)

class Ptychograph(stem4d.Image4d):


    def __init__(self, foldername):
        super().__init__(foldername)
        if foldername[-1] != '/':
            foldername += '/'
        self.foldername = foldername
        self.foldername2 = foldername

        self.fft_image4d =""
        self.aberr_image =""
        self.modes=""
        self.aperture_zero =""
        self.D0 = 100
        self.tifnames=''
        # This value is needed to be adjusted depending on noise level.
        # See for example C.M. O'Leary et. al. Ultramicroscopy 221 (2021)113189.
        self.WDD_epciron = 0.01



    def calc_abs_phase(self,image,wrapped=False):
        abs_image = np.abs(image)
        if wrapped==True:
            phase_image=np.angle(image)
        else:
            phase_image = unwrap_phase(np.angle(image)) #
        return abs_image, phase_image

    def get_strong_g_FFT_image(self, num):

        FFT_abs = Image.open(self.foldername+'fft_abs.tif')
        FFT_abs = np.array(FFT_abs)
        intensity = np.sort(np.unique(FFT_abs))[::-1] #

        for i in range(num):
            intensity = np.sort(np.unique(FFT_abs))[::-1]
            ys, xs = np.where(FFT_abs == intensity[i])

            y = ys[0]
            x = xs[0]
            FFT_abs[y-3:y+4, x-3:x+4] = 0
            G = self.get_FFT_image(y, x, saveimg=True)
            print("G image at=(y,x):(",y,x,")")
            plt.imshow(np.abs(G))
            plt.show()
        return



    def get_FFT_image(self, y, x, saveimg=True):
        name = self.foldername + "FFTs/line_" + str(y)+ ".npy"
        lineimage = np.load(name)
        lineimage = lineimage.transpose(1,2,0)
        image = lineimage[x, :, :]
        if saveimg == True:
            abs_image, phase_image = self.calc_abs_phase(image)
            Image.fromarray(abs_image).save(self.foldername
                                           + 'FFT_abs_image_y='
                                           + str(y)
                                           + '_x='
                                           + str(x)
                                           + '.tif')
            Image.fromarray(phase_image).save(self.foldername
                                             + 'FFT_phase_image_y='
                                             + str(y)
                                             + '_x='
                                             + str(x)
                                             + '.tif')
        return image



    def FFT4D_CPU_Pool(self, cpu_num=os.cpu_count()-5,saveFFT4DData = False):#


        now = datetime.now().time()
        print("*******FFT 4D starts*****************:",now)
        os.makedirs(self.foldername+'FFTs', exist_ok=True)
        abs_image = np.zeros((self.y, self.x))
        image4d = np.zeros((self.y, self.x, self.ky, self.kx//self.bin))
        tifnames = glob.glob(self.foldername + "Lines/*.npy")
        abs_image = np.zeros((self.y, self.x))
        now = datetime.now().time()

        print("*******Reading line data*****************:",now)
        for name in tifnames:
            y = int(name[name.rfind('line_')+5:name.rfind('.npy')])
            lineimage = np.load(name)
            image4d[y, :, :, :] = self.bin_lineimage(lineimage)
        del(lineimage,tifnames)
        now = datetime.now().time()
        print("*******FFT starts*****************:",now)
        fft_image4d = np.fft.fftshift(
                       np.fft.fft(
                       np.fft.fftshift(
                       np.fft.fft(image4d, axis=1), axes=1), axis=0), axes=0)
        del(image4d)

        now = datetime.now().time()
        print("*******FFT finished. Start saving*****************:",now)
        input = [(y,fft_image4d[y,:,:,:]) for y in range(self.y)]
        now = datetime.now().time()
        print("******* input data ready start Pool *****************:",now)
        with Pool(cpu_num) as p:
            p.map(self.Pool_save_fft_images_y, input)
        p.close()
        p.terminate()
        del(input)
        gc.collect()
        now = datetime.now().time()
        print("finish saving fft_ line by line  data:",now)
        now = datetime.now().time()
        print("Start making fft_abs image with Pool:",now)
        with Pool(cpu_num) as p:
            result = p.map(self.Pool_make_fft_abs_image,  range(self.y))
        p.close()
        p.terminate()
        abs_image = np.array(result).reshape((self.y,self.x))
        plt.imshow(np.abs(abs_image))
        plt.show()
        now = datetime.now().time()
        print("finish making fft_abs image with Pool:",now)
        im = Image.fromarray(abs_image.astype(np.float32))
        im.save(self.foldername+'fft_abs.tif')

    def Pool_save_fft_images_y(self, input):
        y,fft_images_y =input
        fft_images_y = fft_images_y.transpose(2,0,1)
        np.save(self.foldername+'FFTs/line_'+str(y),fft_images_y)



    def Pool_make_fft_abs_image(self,  y):
        fft_images_y = np.load(self.foldername+'FFTs/line_'+str(y)+".npy")

        fft_images_y = fft_images_y.transpose(1,2,0)
        abs_images  = np.abs(fft_images_y)
        abs_image_x = abs_images.sum(2).sum(1)
        return abs_image_x

    def measure_center_and_radius(self, y=None, x=None):
        if y == None:
            y = self.y//2
        if x == None:
            x = self.x//2
        image = self.get_FFT_image(y, x)
        thresh = np.zeros((self.ky, self.kx//self.bin))
        thresh[np.where(image >= np.mean(image))] = 1
        Image.fromarray(thresh).save(self.foldername + 'thresh_image.tif')
        cen_ky, cen_kx = ndimage.measurements.center_of_mass(thresh)
        radius = math.sqrt(np.sum(thresh)/np.pi) * self.pixelsize_k
        plt.imshow(thresh)
        plt.show()


        self.cen_kx = cen_kx
        self.cen_ky = cen_ky
        self.radius = radius

        print("cen_ky=",cen_ky," cen_kx=",cen_kx,"radius=",radius)
        return cen_ky, cen_kx, radius


    def calc_distanceratio_and_rotation(self,
                                        diff_ky,
                                        diff_kx,
                                        diff_y,
                                        diff_x,

                                        cen_ky=None,
                                        cen_kx=None,
                                        cen_y=None,
                                        cen_x=None,
                                        num= 5):
        if cen_ky == None:
           cen_ky = self.cen_ky
        if cen_kx == None:
           cen_kx = self.cen_kx
        if cen_y == None:
           cen_y = self.y//2
        if cen_x == None:
           cen_x = self.x//2
        diff1 = math.sqrt((cen_y-diff_y)**2 + (diff_x-cen_x)**2) #
        diff2 = math.sqrt((cen_ky-diff_ky)**2 + (diff_kx-cen_kx)**2)#
        distance_ratio = diff2/diff1
        angle1 = math.atan2(cen_y - diff_y, diff_x-cen_x)
        angle2 = math.atan2(cen_ky - diff_ky, diff_kx-cen_kx)
        rotation = angle2 - angle1
        self.distance_ratio = distance_ratio
        self.rotation = rotation

        print("Acc=",self.accvol," CL=",self.cl," CLA=",self.clap1," mag=",self.mag," x=",self.x," bin=",self.bin)
        print("distance_ratio=",distance_ratio, " rotation[rad]=",rotation)
        print("rotaion[deg]=",rotation*180/np.pi)


        G = self.get_FFT_image(self.y//2,self.x//2)#
        print("G image at=(y,x):(",self.y//2,self.x//2,")")
        plt.imshow(np.abs(G))
        plt.show()
        aperture_circle = self.make_circularfilter_circle(self.cen_ky,self.cen_kx,self.radius)
        plt.imshow(np.abs(G)+ np.max(np.abs(G))*aperture_circle)
        plt.show()
        FFT_abs = hs.load(self.foldername+'fft_abs.tif').data
        FFT_abs[self.y//2-3:self.y//2+4, self.x//2-3:self.x//2+4] = 0 #
        intensity = np.sort(np.unique(FFT_abs))[::-1]
        for i in range(num):
            intensity = np.sort(np.unique(FFT_abs))[::-1]
            ys, xs = np.where(FFT_abs == intensity[i])
            y = ys[0]
            x = xs[0]
            FFT_abs[y-3:y+4, x-3:x+4] = 0
            G = self.get_FFT_image(y, x)
            print("G image at=(y,x):(",y,x,")")
            plt.imshow(np.abs(G))
            plt.show()

            offset_ky, offset_kx = self.calc_offsets(y, x)
            print("y,x, offset y,x=",y,x,offset_ky,offset_kx)
            aperture0 = self.make_circularfilter(self.cen_ky,
                                                 self.cen_kx,
                                                 0,
                                                 self.radius)
            aperture_plus_k = self. make_circularfilter(self.cen_ky + offset_ky,
                                                        self.cen_kx + offset_kx,
                                                        0,
                                                        self.radius)
            aperture_minus_k = self. make_circularfilter(self.cen_ky - offset_ky,
                                                         self.cen_kx - offset_kx,
                                                         0,
                                                         self.radius)
            A_plus = aperture0*aperture_plus_k*np.abs((aperture_minus_k-1))

            aperture0_circle = self.make_circularfilter_circle(self.cen_ky,
                                                 self.cen_kx,
                                                 self.radius)
            aperture_plus_k_circle = self. make_circularfilter_circle(self.cen_ky + offset_ky,
                                                        self.cen_kx + offset_kx,
                                                        self.radius)
            aperture_minus_k_circle = self. make_circularfilter_circle(self.cen_ky - offset_ky,
                                                         self.cen_kx - offset_kx,
                                                         self.radius)
            A_plus_circle = aperture0_circle+aperture_plus_k_circle+aperture_minus_k_circle
            plt.imshow( A_plus_circle*np.max(np.abs(G))+np.abs(G))
            plt.show()
        return distance_ratio, rotation


    def draw_distanceratio_and_rotation(self,   num= 10):
        print("Acc=",self.accvol," CL=",self.cl," CLA=",self.clap1," mag=",self.mag," x=",self.x," bin=",self.bin)
        print("distance_ratio=",self.distance_ratio, " rotation[rad]=",self.rotation)
        G = self.get_FFT_image(self.y//2,self.x//2)
        print("G image at=(y,x):(",self.y//2,self.x//2,")")
        plt.imshow(np.abs(G))
        plt.show()
        aperture_circle = self.make_circularfilter_circle(self.cen_ky,self.cen_kx,self.radius)
        plt.imshow(np.abs(G)+ np.max(np.abs(G))*aperture_circle)
        plt.show()

        FFT_abs = hs.load(self.foldername+'fft_abs.tif').data
        FFT_abs[self.y//2-3:self.y//2+4, self.x//2-3:self.x//2+4] = 0
        intensity = np.sort(np.unique(FFT_abs))[::-1]
        for i in range(num):
            intensity = np.sort(np.unique(FFT_abs))[::-1]
            ys, xs = np.where(FFT_abs == intensity[i])
            y = ys[0]
            x = xs[0]
            FFT_abs[y-3:y+4, x-3:x+4] = 0
            G = self.get_FFT_image(y, x)
            print("G image at=(y,x):(",y,x,")")
            plt.imshow(np.abs(G))
            plt.show()

            offset_ky, offset_kx = self.calc_offsets(y, x)
            print("y,x, offset y,x=",y,x,offset_ky,offset_kx)
            aperture0 = self.make_circularfilter(self.cen_ky,
                                                 self.cen_kx,
                                                 0,
                                                 self.radius)
            aperture_plus_k = self. make_circularfilter(self.cen_ky + offset_ky,
                                                        self.cen_kx + offset_kx,
                                                        0,
                                                        self.radius)
            aperture_minus_k = self. make_circularfilter(self.cen_ky - offset_ky,
                                                         self.cen_kx - offset_kx,
                                                         0,
                                                         self.radius)
            A_plus = aperture0*aperture_plus_k*np.abs((aperture_minus_k-1))

            aperture0_circle = self.make_circularfilter_circle(self.cen_ky,
                                                 self.cen_kx,
                                                 self.radius)
            aperture_plus_k_circle = self. make_circularfilter_circle(self.cen_ky + offset_ky,
                                                        self.cen_kx + offset_kx,
                                                        self.radius)
            aperture_minus_k_circle = self. make_circularfilter_circle(self.cen_ky - offset_ky,
                                                         self.cen_kx - offset_kx,
                                                         self.radius)
            A_plus_circle = aperture0_circle+aperture_plus_k_circle+aperture_minus_k_circle
            plt.imshow( A_plus_circle*np.max(np.abs(G))+np.abs(G))
            plt.show()
        return


    def show_aberration_figure(self, aberrations):
        aberr_image = self.make_aberration_image(0,0,aberrations)
        print("aberr image")
        plt.imshow(np.angle(aberr_image))
        plt.show()

    def calc_offsets(self, y ,x):
        cen_y = self.y//2
        cen_x = self.x//2
        distance = math.sqrt((cen_y - y)**2 + (x - cen_x)**2)*self.distance_ratio
        theta = math.atan2(cen_y - y, x - cen_x) + self.rotation
        offset_ky = -1 * distance * math.sin(theta)
        offset_kx = distance * math.cos(theta)
        return offset_ky, offset_kx

    def CTF(self,filename=""):
        #rotation must be radian
        #radius must be wrriten in mrad
        # ky and kx must be written in integer

        num_of_segment = 2
        seg_i_angle = [0,15]
        seg_o_angle =[0,30]
        seg_azu_angle = np.array([[0, 90], [180, 270]])
        k_CoM = np.array([7.5,15+7.5])


        startTime = time.time()
        CTF = np.zeros((self.y, self.x)).astype(np.complex)

        for qy in range(self.y):
            for qx in range(self.x):
                offset_ky, offset_kx = self.calc_offsets(qy, qx)
                aperture0 = self.make_circularfilter(self.cen_ky,
                                                     self.cen_kx,
                                                     0,
                                                     self.radius)
                aperture_plus_k = self. make_circularfilter(self.cen_ky + offset_ky,
                                                            self.cen_kx + offset_kx,
                                                            0,
                                                            self.radius)
                aperture_minus_k = self. make_circularfilter(self.cen_ky - offset_ky,
                                                             self.cen_kx - offset_kx,
                                                             0,
                                                             self.radius)
                A_plus = aperture0*aperture_plus_k*np.abs((aperture_minus_k-1))
                A_minus = aperture0*aperture_minus_k*np.abs((aperture_plus_k-1))

                for segment_index in range(num_of_segment):
                    D_k = self.make_circularfilter(0,0,seg_i_angle[segment_index],seg_o_angle[segment_index],seg_azu_angle[segment_index])
                    CTF[qy, qx] += np.sum(D_k*A_minus * k_CoM[segment_index]-D_k*A_plus* k_CoM[segment_index])

        Image.fromarray(abs_phi).save(self.foldername + 'CTF_'+ filename + '.tif')
        elapsed_time = time.time()- startTime
        print("elapsed time:{0}".format(elapsed_time))


    def get_phi_from_fft4D(self, qy):

        phi_qy = cp.zeros(self.x).astype(cp.complex)

        name = self.foldername + "FFTs/line_" + str(qy)+ ".npy"
        lineimage = np.load(name)
        lineimage = lineimage.transpose(1,2,0)
        for qx in range(self.x):
            G = lineimage[qx, :, :] #[Qx, ky, kx]の順なので、Gは[ky,kx]
            offset_ky, offset_kx = self.calc_offsets(qy, qx)
            aperture0 = self.make_circularfilter(self.cen_ky,
                                                 self.cen_kx,
                                                 0,
                                                 self.radius)
            aperture_plus_k = self. make_circularfilter(self.cen_ky + offset_ky,
                                                        self.cen_kx + offset_kx,
                                                        0,
                                                        self.radius)
            aperture_minus_k = self. make_circularfilter(self.cen_ky - offset_ky,
                                                         self.cen_kx - offset_kx,
                                                         0,
                                                         self.radius)
            A_plus = aperture0*aperture_plus_k*np.abs((aperture_minus_k-1))
            A_minus = aperture0*aperture_minus_k*np.abs((aperture_plus_k-1))
            phi_qy[qx] = np.sum(G*A_minus-G*A_plus)
            if qy == self.y//2 and qx == self.x//2:
                phi_qy[qx] = np.sum(G*aperture0)

        return phi_qy



    def WDD_multi(self, aberrations=np.zeros(12),filename="",cpu_num=os.cpu_count()-5):
        #rotation must be radian
        #radius must be wrriten in mrad
        # ky and kx must be written in integer

        phi_k = cp.zeros((self.y, self.x)).astype(cp.complex)
        self.aberr_image = self.make_aberration_image(0,0,aberrations)

        self.aperture_zero = self.make_circularfilter(self.cen_ky,
                                                             self.cen_kx,
                                                             0,
                                                             self.radius)

        now = datetime.now().time()
        print("*******WDD starts*****************:",now)

        with Pool(cpu_num) as p:
            result = p.map(self.get_phi_for_WDD, range(self.y))
        p.close()  # add this.
        p.terminate()  # add this.

        phi_k = cp.array(result).reshape((self.y,self.x))

        now = datetime.now().time()
        print("*******Pool calculation finished. FFT by GPU starts*****************:",now)


        phi = cp.conj(phi_k)
        phi = cp.fft.ifft2(cp.fft.ifftshift(phi))
        phi = cp.flip(phi)
        abs_phi, phase_phi = self.calc_abs_phase(cp.asnumpy(phi))
        Image.fromarray(abs_phi).save(self.foldername2 + 'Phi_abs_WDD_multi'+ filename + '.tif')
        Image.fromarray(phase_phi).save(self.foldername2  +'Phi_phase_WDD_multi'+ filename + '.tif')

        plt.figure(figsize=(6,6))
        plt.imshow( phase_phi  )
        plt.show()



    def get_phi_for_WDD(self, qy):
        phi_k_qy = np.zeros(self.x).astype(np.complex)

        name = self.foldername + "FFTs/line_" + str(qy)+ ".npy"
        lineimage = np.load(name)
        lineimage = lineimage.transpose(1,2,0)
        for qx in range(self.x):
            G = lineimage[qx, :, :]
            offset_ky, offset_kx = self.calc_offsets(qy, qx)
            H = np.fft.ifft2(np.fft.ifftshift(G))
            aperture_plus_k = self. make_circularfilter(self.cen_ky + offset_ky,
                                                        self.cen_kx + offset_kx,
                                                        0,
                                                        self.radius)
            probe_zero = self.aberr_image * self.aperture_zero
            probe_plus_k = np.roll(self.aberr_image,
                                   (round(offset_ky), round(offset_kx)),
                                   axis=(0, 1))#offset分、aberr_image画像をスクロールする。
            probe_plus_k *= aperture_plus_k# qy,qxオフセットした位置の収差関数と絞り関数。
            A = probe_zero * np.conj(probe_plus_k)# χ（K)ｘ χ＊（K+Q)
            if np.sum(A) != 0:
                kai_A = np.fft.ifft2(np.fft.ifftshift(A))#
                kai_phi = np.conj(kai_A)*H/(np.abs(kai_A)**2+self.WDD_epciron)
                D = np.fft.fftshift(np.fft.fft2(kai_phi))#
                phi_k_qy[qx] = D[self.ky//2, self.kx//2//self.bin]#
            if qy == self.y//2 and qx == self.x//2:
                self.D0 = np.sqrt(D[self.ky//2, self.kx//2//self.bin])

        return phi_k_qy



    def calc_aberrations(self, y, x):

        aberration = np.array([(x**2 + y**2)/2,
                               (x**2 - y**2)/2,
                               x*y,
                               (x**3 - 3*x*y**2)/3,
                               (3*x**2*y - y**3)/3,
                               (x**3 + x*y**2)/3,
                               (y**3 + x**2*y)/3,
                               (x**4 + y**4 + 2*x**2*y**2)/4,
                               (x**4 + y**4 - 6*x**2*y**2)/4,
                               x**3*y - x*y**3,
                               (x**4 - y**4)/4,
                               (x**3*y + x*y**3)/2])
        aberration *= 2*math.pi/(self.wavelength * 1e+9)
        return aberration



    def measure_aberration_by_specific_Q_iterative(self, num, Qs, cnt_thresh=5):
        aberr_mat = np.zeros((0,12))
        b_array = np.zeros(0)
        y_array = np.zeros((0),dtype=np.int)
        x_array = np.zeros((0),dtype=np.int)
        aberr_array = np.zeros((12))
        beta = np.zeros(12)
        identity_part = np.zeros((0,num))

        offset_ky_kx_list = list()
        Q_K_list = list()
        for i in range(num):
            y = Qs[i][0]
            x = Qs[i][1]


            y_array = np.hstack((y_array, y))
            x_array = np.hstack((x_array, x))
            G = self.get_FFT_image(y, x)
            offset_ky, offset_kx = self.calc_offsets(y, x)
            offset_ky_kx_list.append([y,x,offset_ky,offset_kx])
            overlap = self.make_overlap_region_with_aberration(offset_ky,
                                                               offset_kx,
                                                               np.zeros(12))
            aperture_minus_k = self. make_circularfilter(self.cen_ky - offset_ky,
                                                         self.cen_kx - offset_kx,
                                                         0,
                                                         self.radius)

            A_plus = overlap*np.abs((aperture_minus_k-1))
            kys, kxs = np.where(np.abs(A_plus) != 0)
            Q_K_list.append([y,x,kys,kxs])
            aberr_mat = np.vstack((aberr_mat, self.make_aberr_matrix(offset_ky,
                                                               offset_kx,
                                                               kys,
                                                               kxs)))
            identity_part_tmp = np.zeros((kys.size, num))
            identity_part_tmp[:,i] = 1
            identity_part = np.vstack((identity_part, identity_part_tmp))

            b_array = np.hstack((b_array, self.make_b_array(G, kys, kxs)))

        aberr_mat = np.hstack((aberr_mat, identity_part))
        aberr_mat_inv = self.make_inv_svd(aberr_mat)
        print("aberr_mat.shape=",aberr_mat.shape)


        for c in range(1,13):
            beta[:c] = 0.5
            cnt = 0
            while cnt <= cnt_thresh:
                aberr_array += aberr_mat_inv.dot(b_array.T)[:12]*beta
                b_array = np.zeros(0)
                for Q_K in Q_K_list:
                    y, x, kys,kxs= Q_K
                    G = self.get_FFT_image(y, x, saveimg=False)
                    offset_ky, offset_kx = self.calc_offsets(y, x)
                    overlap = self.make_overlap_region_with_aberration(offset_ky,
                                                                       offset_kx,
                                                                       aberr_array)
                    aperture_minus_k = self. make_circularfilter(self.cen_ky - offset_ky,
                                                                 self.cen_kx - offset_kx,
                                                                 0,
                                                                 self.radius)
                    A_plus = overlap*np.abs((aperture_minus_k-1))
                    G_diff = G*np.conj(A_plus)
                    b_array = np.hstack((b_array, self.make_b_array(G_diff,
                                                                    kys,
                                                                    kxs)))

                overlap = self.make_disk_with_aberration(0,0,  aberr_array)
                abs_overlap, phase_overlap = self.calc_abs_phase(overlap,wrapped=True)
                cnt += 1
        overlap = self.make_disk_with_aberration(0,0,  aberr_array)
        abs_G_diff, phase_G_diff = self.calc_abs_phase(overlap,wrapped=True)

        Image.fromarray(abs_G_diff).save(self.foldername + 'G_diff_abs.tif')
        Image.fromarray(phase_G_diff).save(self.foldername + 'G_diff_phase.tif')
        return aberr_array

    def make_overlap_region_with_aberration(self, dky, dkx, aberr_array):
        disk_zero = self.make_disk_with_aberration(0, 0, aberr_array)
        disk_plus = self.make_disk_with_aberration(dky,
                                                   dkx,
                                                   aberr_array)
        disk_minus = self.make_disk_with_aberration(-dky,
                                                    -dkx,
                                                    aberr_array)
        overlap = disk_zero * np.conj(disk_plus) * np.abs(disk_minus-1)
        overlap -= disk_minus * np.conj(disk_zero) * np.abs(disk_plus-1)
        return overlap

    def make_disk_with_aberration(self, dky, dkx, aberr_array):
        aberr = self.make_aberration_image(dky, dkx, aberr_array, saveimg=False)
        aperture = self.make_circularfilter(self.cen_ky + dky,
                                            self.cen_kx + dkx,
                                            0,
                                            self.radius)
        return aperture * aberr

    def make_aberration_image(self, dky, dkx, aberr_array, saveimg=False):
        aberr_array = np.asarray(aberr_array,dtype="float32")
        aberr_phase = np.zeros((self.ky, self.kx//self.bin))
        for ky in range(self.ky):
            for kx in range(self.kx//self.bin):
                aberr_phase[ky, kx] = self.calc_aberrations((ky-dky-self.cen_ky )* self.pixelsize_k * 0.001,
                                                          (kx-dkx-self.cen_kx )* self.pixelsize_k * 0.001).dot(aberr_array)
        aberr = np.exp(1j * aberr_phase)
        if saveimg == True:
            abs_aberr, phase_aberr = self.calc_abs_phase(aberr)
            Image.fromarray(abs_aberr).save(self.foldername + 'aberr_abs.tif')
            Image.fromarray(phase_aberr).save(self.foldername + 'aberr_phase.tif')
        return aberr

    def make_inv_svd(self, matrix):
        y, x = matrix.shape
        s, v, d = np.linalg.svd(matrix)
        if y > x:
            v = np.hstack((np.diag(1/v), np.zeros((x, y-x))))
        elif y < x:
            v = np.vstack((np.diag(1/v), np.zeros((x-y, y))))
        elif y == x:
            v = np.diag(1/v)
        matrix_inv = d.T.dot(v).dot(s.T)
        return matrix_inv

    def make_b_array(self, G, kys, kxs):
        _, phase = self.calc_abs_phase(G)
        b_array = np.zeros((kys.size))
        ind = 0
        for ky, kx in zip(kys, kxs):
            b_array[ind] = phase[ky, kx]
            ind += 1
        return b_array

    def make_aberr_matrix(self, dky, dkx, kys, kxs):
        aberr_mat_sizem = kys.size
        aberr_mat_sizen = 12# 収差係数は12個。
        aberr_mat = np.zeros((aberr_mat_sizem, aberr_mat_sizen))
        ind = 0
        for ky, kx in zip(kys, kxs):
            aberr_mat[ind, :] = self.calc_aberrations((ky-self.cen_ky) * self.pixelsize_k * 0.001,
                                                    (kx-self.cen_kx) *  self.pixelsize_k * 0.001)
            aberr_mat[ind, :] -= self.calc_aberrations((ky-dky-self.cen_ky) *  self.pixelsize_k * 0.001,
                                                     (kx-dkx-self.cen_kx) * self.pixelsize_k * 0.001)
            ind += 1
        return aberr_mat
