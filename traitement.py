import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import time
from tqdm import tqdm
# On importe Tkinter
from tkinter import ttk
import scipy.misc        as sm

import numpy as np
from PIL                 import Image as Img
from PIL                 import Image,ImageTk
from tkinter import filedialog
import cv2
from tkinter import *
import sys
from tkinter.messagebox import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from numpy.fft import fft, ifft, fft2, ifft2
import  utils

from PIL import Image as Img
import matplotlib.cm as cm
import math
import os
import scipy.ndimage
import scipy
import tkinter.filedialog

# On crée une fenêtre, racine de notre interface

#if 0 alors imgGrais
#if 1 alors imgColor
#if -1 alors imgNoChange
fenetre = Tk()
fenetre.title("TI")
fenetre.geometry("1000x600")
canvas=Canvas(fenetre,width=1000,height=1000,background="grey")
canvas.pack()
#'#4056A4'


    
    

#########################""
mylabel = canvas.create_text((225, 420), text="Image originale",fill="white")
Item1=canvas.create_rectangle(50,50,400,400,outline="black",fill="white")
mylabel = canvas.create_text((675, 420), text="Image traitée",fill="white")
Item2=canvas.create_rectangle(500,50,850,400,outline="black",fill="white")
text = Label(fenetre)
text.place(x=500,y=50)

l=Label(fenetre)
l.place(x=50, y=50)

#image = Image.open(file)
#image = tkinter.PhotoImage(file =file)
#img = ImageTk.PhotoImage(image.resize((200, 200))) 
#img = ImageTk.PhotoImage(Image.open().resize(200, 200)) 
#gif1 = PhotoImage(file='/home/ennaji/Bureau/Lenna.png',height=200,  width=200 )



'''
img = ImageTk.PhotoImage(Image.open('/home/ennaji/Bureau/Lenna.png').resize(200, 200)) # the one-liner I used in my app
label = Label(root, image=img)
label.image = img # this feels redundant but the image didn't show up without it in my app
label.pack()
'''


'''
cannevas = Canvas(canvas,height=300,width=350,background='blue')
cannevas.place(x = 600, y = 100, width=250, height=250)
cannevas.create_image(0,0, anchor=NW, image=img)
'''


def ouvrir():
    global filepath
    filepath =filedialog.askopenfilename(initialdir="/",title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
    
    img=cv2.imread(filepath,1)
    img = cv2.resize(img,(400,350))
    img = cv2.cvtColor( img,cv2.COLOR_RGB2BGR)
    
    img = Img.fromarray(img)
    img=ImageTk.PhotoImage(img)
    l['image']=img
    
    
    l.photo = img
    

#################################################################
def Seuillage3d():
    ##filepath=ouvrir()
    img = cv2.imread(filepath,1)  
    img = cv2.resize(img,(400,350))
    #img = np.where( A > 147,0,255)
    img.setflags(write=1)
    temp=np.zeros(shape=(img.shape[0],img.shape[1]))
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            for k in range(0,img.shape[2]):
                if(img[i][j][k]>=147):
                    img[i][j][k]=255
                else:
                    img[i][j][k]=0
    img = Img.fromarray(img)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
   
#################################################################
def  colorInversion():
     pixels = cv2.imread(filepath,1)  
     pixels = cv2.resize(pixels,(400,350))
     for i in range(pixels.shape[0]):
          for j in range(pixels.shape[1]):
               x,y,z = pixels[i,j][0],pixels[i,j][1],pixels[i,j][2]
               x,y,z = abs(x-255), abs(y-255), abs(z-255)
               pixels[i,j] = (x,y,z)
     img = cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR)
     img = Img.fromarray(img)
     img = ImageTk.PhotoImage(img)
     text['image']=img
     text.photo=img
    
   
#################################################################
def mat2gray(img):
    A = np.double(img)
    out = np.zeros(A.shape, np.double)
    normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)
    return out


def gaussianNoise():
     img = cv2.imread(filepath,1) 
     img = cv2.resize(img,(400,350))  
     img=mat2gray(img)
     row,col,ch= img.shape
     mean = 0
     var = 0.1
     sigma = var**0.9
     gauss = np.random.normal(mean,sigma,img.shape)
     gauss = gauss.reshape(row,col,ch)
     noisyImage = img + gauss
     ##result.setPillImage(Image.fromarray((noisyImage * 255).astype(np.uint8)))
     img = Img.fromarray(cv2.cvtColor((noisyImage*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
     img = ImageTk.PhotoImage(img)
     text['image']=img
     text.photo=img
#################################################################

def NG():
    ##filepath=ouvrir()
    img = cv2.imread(filepath,1) 
    img = cv2.resize(img,(400,350))  
    img = Img.fromarray(img)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
    
    
##############################################################
def  grayScaleImage():
     img = cv2.imread(filepath,1) 
     img = cv2.resize(img,(400,350))  
     grayImage = np.zeros(img.shape)
     R = np.array(img[:, :, 0])
     G = np.array(img[:, :, 1])
     B = np.array(img[:, :, 2])

     R = (R *.299)
     G = (G *.587)
     B = (B *.114)

     Avg = (R+G+B)
     grayImage = img

     for i in range(3):
          grayImage[:,:,i] = Avg
          
     img = Img.fromarray(grayImage)
     img = ImageTk.PhotoImage(img)
     text['image']=img
     text.photo=img
#########################################################
def  poivreAndSelNoise():
     img = cv2.imread(filepath,1) 
     img = cv2.resize(img,(400,350))  
     # Getting the dimensions of the image
     row , col, ch = img.shape
      
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
     number_of_pixels = random.randint(300, 10000)
     for i in range(number_of_pixels):
        
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to white
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixel
     number_of_pixels = random.randint(300 , 10000)
     for i in range(number_of_pixels):
        
          y_coord=random.randint(0, row - 1)
          
          x_coord=random.randint(0, col - 1)
          
          img[y_coord][x_coord] = 0
     img = Img.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
     img = ImageTk.PhotoImage(img)
     text['image']=img
     text.photo=img
#########################################################
    
def sp_noise():
    '''
Add salt and pepper noise to image
prob: Probability of the noise
    '''
   
    img = cv2.imread(filepath,1) 
    img = cv2.resize(img,(400,350))
    prob=0.05
    output = np.zeros(img.shape,np.uint8)
    thres = 1 - prob 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    #return output
    ##result.setPillImage(Img.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))) 
    img = Img.fromarray(Img.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)))
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img

#image = cv2.imread("/home/ennaji/Master/Tratement d'image/Test/Ennaji.png",0) # Only for grayscale image
#noise_img = sp_noise(image,0.05)
#########################################################
def FiltreMedian():
    
    img = cv2.imread(filepath,1)
    img = cv2.resize(img,(400,350))
    ##img=img.tolist()
    
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            temp = []
            temp.clear()
           ## img[i][j]=img[i][j].tolist()
            ## img[i-1][j-1]+img[i-1][j]+img[i-1][j+1]+img[i][j-1] img[i][j] img[i][j+1]
            ## img[i+1][j-1] img[i+1][j] img[i+1][j+1]
            #print("hhhhh")
            #print((img[i][j])[0])
            temp=[(img[i-1][j-1])[0],(img[i-1][j])[0],(img[i-1][j+1])[0],(img[i][j-1])[0] ,(img[i][j])[0] ,(img[i][j+1])[0], (img[i+1][j-1])[0] ,(img[i+1][j])[0] ,(img[i+1][j+1])[0]]
           
            """temp.append((img[i-1][j])[0])
            ##temp[1]= img[i-1][j]
            
            ##temp[2]= img[i-1][j+1]
            temp.append((img[i-1][j+1])[0])
            ##temp[3]= img[i][j-1]
            temp.append((img[i][j-1])[0])
            ##temp[4]= img[i][j]
            temp.append((img[i][j])[0])
            ##temp[5]= img[i][j+1]
            temp.append((img[i][j+1])[0])
            ##temp[6]= img[i+1][j-1]
            temp.append((img[i+1][j-1])[0])
            ##temp[7]= img[i+1][j]
            temp.append((img[i+1][j])[0])
            ##temp[8]= img[i+1][j+1]
            temp.append((img[i+1][j+1])[0])
            ##print(temp[j])
           ## print(img[i][j])
            ##print(Trier(temp))"""
            a=trier(temp)
           
            print("////////////////////////////////////////")
            print(img[i][j])
            
            img[i][j]=a
            print(img[i][j])
            ##img[i][j]=np.array(img[i][j])
           ## img[i][j]=Trier(temp)
            ##print("#################")
            ##print(img[i][j])
    
    img = Img.fromarray(img)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
#########################################################
def trier(arr):
    for i in range(len(arr) - 1, 0, -1):
        for j in range(i):
            if arr[j] > arr[j + 1]:
                arr[j + 1], arr[j] = arr[j], arr[j + 1]
    return arr[(4)]
#########################################################          
def Trier(t):
    for i in range(0,len(t)-1):
        for j in range(1,len(t)):
            if(t[i]>t[j]):
                f=t[i]
                t[i]=t[j]
                t[j]=f
    return t[(len(t)/2)+1]

######################################
def median_filter1():
    data = cv2.imread(filepath,1)
    data = data.getOimagePil()
    #data=cv2.cvtColor(np.array(data), cv2.COLOR_RGB2BGR)
    #data = cv2.resize(img,(400,350))
    temp = []
    filter_size=3
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])
                            print(temp[k])

            
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    data_final=cv2.resize(data_final,(500,400))
    img = ImageTk.PhotoImage(Image.fromarray((data_final * 255).astype(np.uint8)))
    text['image']=img
    text.photo=img
    return data_final

#########################################################
def median_filter():   
    ##filepath=ouvrir()
    img = cv2.imread(filepath,1) 
    img = cv2.resize(img,(400,350))
    img = cv2.cvtColor( img,cv2.COLOR_RGB2BGR)
    
    #imgF=np.zeros(256)
    newImg = cv2.medianBlur(img,5)
    #newImg= gaussian_filter(img, sigma=3)
    img = Img.fromarray(newImg)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img

#########################################################
def FGaussien():   
    ##filepath=ouvrir()
    img = cv2.imread(filepath,1) 
    img = cv2.resize(img,(400,350))
    
    img = cv2.cvtColor( img,cv2.COLOR_RGB2BGR)
    #imgF=np.zeros(256)
    newImg = cv2.GaussianBlur(img, (5, 5), 3)
    #newImg= gaussian_filter(img, sigma=3)
    img = Img.fromarray(newImg)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
#########################################################
def pyramidalFilter():
        mask=np.array([[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [3, 6, 9, 6, 3], [2, 4, 6, 4, 2], [1, 2, 3, 2, 1]])
        mask = (1/81)*mask; 
        img = cv2.imread(filepath,1) 
        img = cv2.resize(img,(400,350))
        img = cv2.cvtColor( img,cv2.COLOR_RGB2BGR)
        img_new=utils.convolve2DRGB(img,mask,padding=0,strides=1)
        img_new=mat2gray(img_new);
        
        #img_tk1 = outils.img_fromCv_toTk (image)
        img = Img.fromarray((img_new * 255).astype(np.uint8))
        
        img = ImageTk.PhotoImage(img)
        text['image']=img
        text.photo=img
#########################################################  
def coniqueFilter():

        mask=np.array([[0, 0, 1, 0, 0], [0, 2, 2, 2, 0], [1, 2, 5, 2, 1], [0, 2, 2, 2, 0], [0, 0, 1, 0, 0]])
        mask = (1/25)*mask; 
        img = cv2.imread(filepath,1) 
        img = cv2.resize(img,(400,350))
        img = cv2.cvtColor( img,cv2.COLOR_RGB2BGR)
        img_new=utils.convolve2DRGB(img,mask,padding=0,strides=1)
        img_new=mat2gray(img_new);
        img = Img.fromarray((img_new * 255).astype(np.uint8))
      
        img = ImageTk.PhotoImage(img)
        text['image']=img
        text.photo=img
#########################################################
def contrast():
    image = cv2.imread(filepath,1) 
    #image = cv2.resize(img,(400,350))
    # Reading the original image, here 0 implies that image is read as grayscale


# Generating the histogram of the original image
    hist,bins = np.histogram(image.flatten(),256,[0,256])

# Generating the cumulative distribution function of the original image
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    ##image=Img.open(filepath)
    #image=np.array(image)
    # Create an empty array to store the final output
   
    image_equalized = cv2.hist_equalized(imgage)

# Generating the histogram of the equalized image
    hist_equalized,bins_equalized = np.histogram(image_equalized.flatten(),256,[0,256])

# Generating the cumulative distribution function of the original image
    cdf_equalized = hist_equalized.cumsum()
    cdf_equalized_normalized = cdf_equalized * hist_equalized.max()/ cdf_equalized.max()
    image_yuv = cv2.cvtColor(image_c, cv2.COLOR_BGR2YUV)

# Applying Histogram Equalization on the original imageof the Y channel
    image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
 

# Apply Min-Max Contrasting

    img = Img.fromarray(image_yuv)
    img = ImageTk.PhotoImage(image)
    text['image']=img
    text.photo=img
#########################################################
def hist():
    ##filepath=ouvrir()
    img = cv2.imread(filepath,1) 
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    #plt.show()
    plt.savefig ( "histogramme.png" )
    s="D:/MIDVI/traitement image/histogramme.png"
    img = cv2.imread(s,cv2.IMREAD_COLOR)
    img= cv2.resize(img,(400,350))
    img = Img.fromarray(img)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img

#########################################################
def ImageMoyenne():
    ##filepath=ouvrir()
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))
    #CettefonctionPermetDeConvertirANiveau de gris
    img.setflags(write=1)
    temp=np.zeros(shape=(img.shape[0],img.shape[1]))
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            for k in range(0,img.shape[2]):
                temp[i][j]+=img[i][j][k]
                
    for i in range(0,temp.shape[0]):
        for j in range(0,temp.shape[1]):
            temp[i][j]/=3
    img = Img.fromarray(temp)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
            
#########################################################
def save():
    f=filedialog.asksaveasfile(initialfile="save.png",initialdir="/",
    title="Enregistrer sous ... une image",
    filetypes=[('png files','.png'),('all files','.*')]
        
    )
#########################################################
def gradientM():
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))
    structurant_size = 5 
    data =  cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #Cross-shaped Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(structurant_size,structurant_size))
    dil = cv2.dilate(data,kernel,iterations = 2)
    ero = cv2.erode(data,kernel,iterations = 1)
    img_final = dil - ero
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    img = Img.fromarray(img_final)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img 

#########################################################  
def whiteTopHat():
    
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))
    structurant_size = np.ones((5,5), np.uint8) 
    data =  cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Cross-shaped Kernel
    kernel =  cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))
    img_final = data - cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel,iterations=2)
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    img = Img.fromarray(img_final)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img 
#########################################################
def blackTopHat():
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))
    structurant_size= np.ones((5,5), np.uint8) 
    data =  cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #Cross-shaped Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))
    img_final = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel,iterations=2) - data
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    img = Img.fromarray(img_final)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img 
#########################################################
def fermeture():
   ## filepath=ouvrir()
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))
    kernel = np.ones((5,5), np.uint8) 
    img_erosion = cv2.erode(img, kernel, iterations=1) 
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1) 
    #img_final = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations=1)
    #img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    #img = Img.fromarray(img_final)
    img_dilation = cv2.cvtColor( img_dilation,cv2.COLOR_RGB2BGR)
    img = Img.fromarray(img_dilation)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
#########################################################    
def ouverture():
   ## filepath=ouvrir()
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))
    kernel = np.ones((5,5), np.uint8) 
    img_dilation = cv2.dilate(img, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    img_erosion = cv2.cvtColor( img_erosion,cv2.COLOR_RGB2BGR)
    
    img = Img.fromarray(img_erosion)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
    
#########################################################    
def Erosion():
    ##filepath=ouvrir()
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))

    # Taking a matrix of size 5 as the kernel 
    kernel = np.ones((5,5), np.uint8) 
    img_erosion = cv2.erode(img, kernel, iterations=1) 
    img_erosion = cv2.cvtColor( img_erosion,cv2.COLOR_RGB2BGR)
    img = Img.fromarray(img_erosion)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
#########################################################
def Delation():
   ## filepath=ouvrir()
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))

    # Taking a matrix of size 5 as the kernel 
    kernel = np.ones((5,5), np.uint8)  
    #img_erosion = cv2.erode(img, kernel, iterations=1) 
    img_dilation = cv2.dilate(img, kernel, iterations=1) 
    img_dilation = cv2.cvtColor( img_dilation,cv2.COLOR_RGB2BGR)
    img = Img.fromarray(img_dilation)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img 
#########################################################    
def decideKernel(leftSetting):
    kernel  = leftSetting.getKernel()
    if kernel=="3X3":
        return 3
    elif kernel =="5X5":
        return 5
    elif kernel=="9X9":
        return 9


    

######################################################################
def contourDetectionSobel():
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))
    data = cv2.GaussianBlur(img, (3, 3), 0)
    kernel_size=3
    # cvsobel = cv2.Sobel(data,cv2.CV_64F,1,0,ksize=3)+cv2.Sobel(data,cv2.CV_64F,0,1,ksize=3)
    # cv2.imshow("cv2 sobel",cv2.cvtColor(np.uint8(np.absolute(cvsobel)),cv2.COLOR_RGB2GRAY))
    (sobel_x,sobel_y) = utils.sobelKernel(kernel_size)
    img_final = utils.convolve2DRGB_float64(img,sobel_x) + utils.convolve2DRGB_float64(data,sobel_y)
    img_final = cv2.cvtColor(np.uint8(np.absolute(img_final)),cv2.COLOR_RGB2GRAY)
    img = Img.fromarray(img_final)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img 
    
#########################################################
def contourDetectionGradient():
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))
    data = cv2.GaussianBlur(img, (3, 3), 0)
    img_final = utils.convolve2DRGB_float64(img,np.array([[-1,0,1]])) + utils.convolve2DRGB_float64(img,np.array([[-1,0,1]]).T)
    img_final = cv2.cvtColor(np.uint8(np.absolute(img_final)),cv2.COLOR_RGB2GRAY)
    img = Img.fromarray(img_final)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img 
#########################################################
def contourDetectionPrewitt():
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))
    data = cv2.GaussianBlur(img, (3, 3), 0)
    kernel_size=3
    (perwitt_x,perwitt_y) = utils.perwittKernel(kernel_size)
    img_final = utils.convolve2DRGB_float64(img,perwitt_x) + utils.convolve2DRGB_float64(img,perwitt_y)
    img_final = cv2.cvtColor(np.uint8(np.absolute(img_final)),cv2.COLOR_RGB2GRAY)
    img = Img.fromarray(img_final)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img 

#########################################################
def contourDetectionRoberts():
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))
    data = cv2.GaussianBlur(img, (3, 3), 0)
    kernel_size=5
    (robert_x,robert_y) = utils.robertKernel(kernel_size)
    img_final = utils.convolve2DRGB_float64(img,robert_x) + utils.convolve2DRGB_float64(img,robert_y)
    img_final = cv2.cvtColor(np.uint8(np.absolute(img_final)),cv2.COLOR_RGB2GRAY)
    img = Img.fromarray(img_final)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img 
#########################################################    
def contourDetectionLaplacian_gaussian():
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))
    data = cv2.GaussianBlur(img, (3, 3), 0)
    kernel_size=5
    log_kernel = utils.logkernel(kernel_size)
    img_final = utils.convolve2DRGB_float64(img,log_kernel) 
    img_final = cv2.cvtColor(np.uint8(np.absolute(img_final)),cv2.COLOR_RGB2GRAY)
    img = Img.fromarray(img_final)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img 
#####################################################
def medianFilter():
        
        data=Img.open(filepath)
        data=np.array(data)
        temp = []
        filter_size=5
        indexer = filter_size // 2
        data_final = []
        data_final = np.zeros((len(data),len(data[0])))
        for i in range(len(data)):
    
            for j in range(len(data[0])):
    
                for z in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                        for c in range(filter_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(filter_size):
                                temp.append(data[i + z - indexer][j + k - indexer])
    
                temp.sort()
                data_final[i][j] = temp[len(temp) // 2]
                temp = []
        print(type(data_final))
        data_final=cv2.resize(data_final,(400,350))
        #img_tk1 = ImageTk.PhotoImage(Image.fromarray((data_final * 255).astype(np.uint8)))
        #img_tk1 = outils.img_fromCv_toTk (image)
        img = Img.fromarray((data_final).astype(np.uint8))
        img = ImageTk.PhotoImage(img)
        text['image']=img
        text.photo=img 

#########################################################
def InverseParColon():
   ## filepath=ouvrir()
    img = cv2.imread(filepath,1)
    img= cv2.resize(img,(400,350))
    for i in range(0,img.shape[0]):
        for j in range(0,int(img.shape[1]/2)+1):
            t=img[i][j]
            img[i][j]=img[i][img.shape[1]-1-j]
            img[i][img.shape[1]-j-1]=t
    img = Img.fromarray(img)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
#########################################################
def InverseParLinge():
    ##filepath=ouvrir()
    img = cv2.imread(filepath,1)
    img= cv2.resize(img,(400,350))
    for i in range(0,int(img.shape[0]/2)+1):
        for j in range(0,img.shape[1]):
            t=img[i][j]
            img[i][j]=img[img.shape[0]-i-1][j]
            img[img.shape[0]-i-1][j]=t
    #return img
    img = Img.fromarray(img)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
#########################################################
def gradient():
    ## filepath=ouvrir()
    img = cv2.imread(filepath,1)
    img= cv2.resize(img,(400,350))
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    img = cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize = 1) 
    img = Img.fromarray(img)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
#########################################################
def laplacian():
    ## filepath=ouvrir()
    img = cv2.imread(filepath,1)
    img= cv2.resize(img,(400,350))
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    #img = cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize = 1) 
    img = Img.fromarray(img)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
#########################################################
def tf2d():
   ## filepath=ouvrir()
    img = cv2.imread(filepath,1) 
    img= cv2.resize(img,(400,350))
    #B=imread('barbara.png')
    img=showfft2(log(abs(fft2(img))))
    title("TF2D de l'image de Barbara")
    img = Img.fromarray(img)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
  
def _quit():
    question=askquestion("Fermer ...","voulez-vous vraiment fermer cette page ?")
    if question=="yes":
        fenetre.quit()     # stops mainloop
        fenetre.destroy() 
#########################################################
def Susan():
    img = cv2.imread(filepath,1)
    img= cv2.resize(img,(400,350))
    img = cv2.cvtColor( img,cv2.COLOR_RGB2BGR)
    r = 3
    raduis = utils.reduceReduis(r) # reduce the raduis to 2 3 5
    
    data_output = utils.add_padding(img, raduis, 0)
   
    data = cv2.cvtColor(data_output,cv2.COLOR_RGB2GRAY)
    data = cv2.medianBlur(data,3) ### to denoise the image 
    data = cv2.GaussianBlur(data,(5,5),0) ### to smooth the image 
    data = data.astype(np.float64)
    nucleus = utils.susanNucleus(raduis)
    g = utils.susanSum(nucleus)//2 ### 3*utils.susanSum(nucleus)/4
    for i in range(raduis,data.shape[0]-raduis):
        for j in range(raduis,data.shape[1]-raduis):
            ir=np.array(data[i-raduis:i+raduis+1, j-raduis:j+raduis+1])
            ir =  ir[nucleus==1]
            ir0 = data[i,j]
            n=np.sum(np.exp(-((ir-ir0)/10)**6)) ## t = 10 
            if n<=g: ## if n>g means this is an homog erea
                n=g-n
            else:
                n=0
            ## we could test only if n != 0 so we capture all edges pixels
            ## instead we  make sure that is a truly an edge
            if n < g//2 and n > 0: 
                ## so give it green color
                data_output[i-1:i+1,j-1:j+1] = (0,0,255)
            
    ##################### remove the padding #################
    img_final = data_output[raduis+2:data_output.shape[0]-raduis-2,raduis+2:data_output.shape[1]-raduis-2]
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    img = Img.fromarray(img_final)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img
#########################################################
def harris():
    img = cv2.imread(filepath,1)
    img= cv2.resize(img,(400,350))
    
    
    k = 0.04
    threshold = 60000
    offset = int(5/2)
    
    
    data_output = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    data_output = cv2.GaussianBlur(data_output,(5,5),0) ### to smooth the image 
    dy, dx = np.gradient(data_output)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    for y in range(offset, img.shape[0]-offset):
        for x in range(offset, img.shape[1]-offset):
            
            #Values of sliding window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1
            
            windowIxx = Ixx[start_y : end_y, start_x : end_x]
            windowIxy = Ixy[start_y : end_y, start_x : end_x]
            windowIyy = Iyy[start_y : end_y, start_x : end_x]
            
            #Sum of squares of intensities of partial derevatives 
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            
            #Calculate r for Harris Corner equation
            r = det - k*(trace**2)

            if r > threshold:
                img[y,x] = (0,0,255)

    img_final = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img = Img.fromarray(img_final)
    img = ImageTk.PhotoImage(img)
    text['image']=img
    text.photo=img


#######################################  frequantiel ########################

def contourDetectionFFT():
    img1 = cv2.imread(filepath,1)
    data= cv2.resize(img1,(400,350))
    mask_size = 3
    
    if data.shape[2] == 3:
        data = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
    data = cv2.GaussianBlur(data, (3, 3), 0)

    ### construct a high pass mask  ######### 
    rows , cols = data.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # center
    mask = np.ones((rows, cols),dtype=np.uint8)
    r = mask_size*6
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    ########### visulize the mask #########
    cv2.imshow("mask",mask*255)

    img = np.fft.fft2(data)
    img = np.fft.fftshift(img)
    img = img * mask
    img_final = np.fft.ifft2(img)
    img_final = np.uint8(np.absolute(img_final))
    img1 = Img.fromarray(img_final)
    img1 = ImageTk.PhotoImage(img1)
    text['image']=img1
    text.photo=img1
#########################################################
def imageEnhancementFFT():
    img1 = cv2.imread(filepath,1)
    data= cv2.resize(img1,(400,350))
    mask_size = 3
    img_final = ""

    ## construct a mask #
    sigmax, sigmay = (mask_size-1)*14, (mask_size-1)*14
    cy, cx = data.shape[0]/2,  data.shape[1]/2
    x = np.linspace(0, data.shape[1], data.shape[1])
    y = np.linspace(0, data.shape[0], data.shape[0])
    X, Y = np.meshgrid(x, y)
    gmask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))
    cv2.imshow("mask",gmask)
    if data.shape[2] == 3:

        r,g,b = cv2.split(data)

        r_fft = np.fft.fft2(r)
        g_fft = np.fft.fft2(g)
        b_fft = np.fft.fft2(b)
        
        ##### shifted #####
        r_fft = np.fft.fftshift(r_fft)
        g_fft = np.fft.fftshift(g_fft)
        b_fft = np.fft.fftshift(b_fft)

        ## apply the mask to each channel
        r_fft = r_fft * gmask
        g_fft = g_fft * gmask
        b_fft = b_fft * gmask
        # cv2.imshow("r",np.uint8(np.absolute(r_fft)))
        # cv2.imshow("g",np.uint8(np.absolute(g_fft)))
        # cv2.imshow("b",np.uint8(np.absolute(b_fft)))
        ## ifft inverse fft ##
        r_fft = np.fft.ifft2(r_fft)
        g_fft = np.fft.ifft2(g_fft)
        b_fft = np.fft.ifft2(b_fft)
        
        img_final = cv2.merge((np.uint8(np.absolute(r_fft)),np.uint8(np.absolute(g_fft)),np.uint8(np.absolute(b_fft))))
        ## BGR sometimes in tkinter is the RGB in CV2 
        # Strange conversion needed!!! 
        img_final = cv2.cvtColor(img_final,cv2.COLOR_BGR2RGB)
    else:
        img = np.fft.fft2(data)
        img = np.fft.fftshift(img)
        img = img * gmask
        img = np.fft.ifft2(img)
        img_final = img
    img1 = Img.fromarray(img_final)
    img1 = ImageTk.PhotoImage(img1)
    text['image']=img1
    text.photo=img1
#########################################################
def imageEnhancementFFTBandPass():
    img1 = cv2.imread(filepath,1)
    data= cv2.resize(img1,(400,350))
    mask_size = 3
    

    rows , cols = data.shape[0], data.shape[1]
    crow, ccol = int(rows / 2), int(cols / 2)  # center
    gmask = np.zeros((rows, cols))
    r_out = mask_size*18
    r_in = mask_size/2 
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    gmask[mask_area] = 1
    ########### visulize the mask #########
    cv2.imshow("mask",gmask*255)
    
    if data.shape[2] == 3:
        
        r,g,b = cv2.split(data)

        r_fft = np.fft.fft2(r)
        g_fft = np.fft.fft2(g)
        b_fft = np.fft.fft2(b)
        
        ##### shifted #####
        r_fft = np.fft.fftshift(r_fft)
        g_fft = np.fft.fftshift(g_fft)
        b_fft = np.fft.fftshift(b_fft)

        ## apply the mask to each channel
        r_fft = r_fft * gmask
        g_fft = g_fft * gmask
        b_fft = b_fft * gmask
        # cv2.imshow("r",np.uint8(np.absolute(r_fft)))
        # cv2.imshow("g",np.uint8(np.absolute(g_fft)))
        # cv2.imshow("b",np.uint8(np.absolute(b_fft)))
        ## ifft inverse fft ##
        r_fft = np.fft.ifft2(r_fft)
        g_fft = np.fft.ifft2(g_fft)
        b_fft = np.fft.ifft2(b_fft)
        
        img_final = cv2.merge((np.uint8(np.absolute(r_fft)),np.uint8(np.absolute(g_fft)),np.uint8(np.absolute(b_fft))))
        ## BGR sometimes in tkinter is the RGB in CV2 
        # Strange conversion needed!!! 
        img_final = cv2.cvtColor(img_final,cv2.COLOR_BGR2RGB)
    else:
        img = np.fft.fft2(data)
        img = np.fft.fftshift(img)
        img = img * gmask
        img = np.fft.ifft2(img)
        img_final = img
    
    img_final = np.uint8(np.absolute(img_final))
    img_final = cv2.cvtColor(img_final,cv2.COLOR_RGB2BGR)
    img1 = Img.fromarray(img_final)
    img1 = ImageTk.PhotoImage(img1)
    text['image']=img1
    text.photo=img1
#########################################################
def contourDetectionButterworth():
    img1 = cv2.imread(filepath,1)
    data= cv2.resize(img1,(400,350))
    mask_size = 3
    if data.shape[2] == 3:
        data = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
    data = cv2.GaussianBlur(data, (3, 3), 0)
    n = 2
    D0 = mask_size * 3
    mask = np.zeros(data.shape[:2])
    rows, cols = data.shape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            mask[y,x] = 1-1/(1+(utils.distance((y,x),center)/D0)**(2*n))
    cv2.imshow("mask",mask)
    img = np.fft.fft2(data)
    img = np.fft.fftshift(img)
    img = img * mask
    img = np.fft.ifft2(img)
    img_final = img

    img_final = np.uint8(np.absolute(img))
    img1 = Img.fromarray(img_final)
    img1 = ImageTk.PhotoImage(img1)
    text['image']=img1
    text.photo=img1
#########################################################
def ButterworthLowPass():
    img1 = cv2.imread(filepath,1)
    data= cv2.resize(img1,(400,350))
    mask_size = 3
    n = 2
    D0 = mask_size * 12
    gmask = np.zeros(data.shape[:2])
    rows, cols = data.shape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            gmask[y,x] = 1/(1+(utils.distance((y,x),center)/D0)**(2*n))
    cv2.imshow("mask",gmask)
    
    if data.shape[2] == 3:
        
        r,g,b = cv2.split(data)

        r_fft = np.fft.fft2(r)
        g_fft = np.fft.fft2(g)
        b_fft = np.fft.fft2(b)
        
        ##### shifted #####
        r_fft = np.fft.fftshift(r_fft)
        g_fft = np.fft.fftshift(g_fft)
        b_fft = np.fft.fftshift(b_fft)

        ## apply the mask to each channel
        r_fft = r_fft * gmask
        g_fft = g_fft * gmask
        b_fft = b_fft * gmask
        # cv2.imshow("r",np.uint8(np.absolute(r_fft)))
        # cv2.imshow("g",np.uint8(np.absolute(g_fft)))
        # cv2.imshow("b",np.uint8(np.absolute(b_fft)))
        ## ifft inverse fft ##
        r_fft = np.fft.ifft2(r_fft)
        g_fft = np.fft.ifft2(g_fft)
        b_fft = np.fft.ifft2(b_fft)
        
        img_final = cv2.merge((np.uint8(np.absolute(r_fft)),np.uint8(np.absolute(g_fft)),np.uint8(np.absolute(b_fft))))
        ## BGR sometimes in tkinter is the RGB in CV2 
        # Strange conversion needed!!! 
        img_final = cv2.cvtColor(img_final,cv2.COLOR_BGR2RGB)
    else:
        img = np.fft.fft2(data)
        img = np.fft.fftshift(img)
        img = img * gmask
        img = np.fft.ifft2(img)
        img_final = img
    
    #img_final = np.uint8(np.absolute(img_final))
    showInPlot(img,fshift_mask_mag, img_final)
    img1 = Img.fromarray(img_final)
    img1 = ImageTk.PhotoImage(img1)
    text['image']=img1
    text.photo=img1
#########################################################
def fft(img):
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        return dft_shift

#########################################################
def bondPass():
        img = cv2.imread(filepath,1)
        img= cv2.resize(img,(400,350))
        dft_shift=fft(img)
        rows, cols = img.shape[0],img.shape[1]
        crow, ccol = int(rows / 2), int(cols / 2)

        mask = np.zeros((rows, cols, 2), np.uint8)
        r_out = 100
        r_in =  50
        center = [crow, ccol]
        x, y = np.ogrid [:rows, :cols]
        mask_area = np.logical_and(((x - center [0]) ** 2 + (y - center [1]) ** 2 >= r_in ** 2),
                                   ((x - center [0]) ** 2 + (y - center [1]) ** 2 <= r_out ** 2))
        mask [mask_area] = 1

        fshift = dft_shift * mask
        fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift [:, :, 0], fshift [:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back [:, :, 0], img_back [:, :, 1])
        showInPlot(img,fshift_mask_mag,img_back)
        img1 = Img.fromarray(img_back)
        img1 = ImageTk.PhotoImage(img1)
        text['image']=img1
        text.photo=img1
#########################################################
def showInPlot(img,fshift_mask_mag,img_back):
    
    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(magnitudeSpec(img), cmap='gray')
    plt.title('After FFT'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(fshift_mask_mag, cmap='gray')
    plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(img_back, cmap='gray')
    plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
    plt.show()



menubar = Menu(fenetre) 
fenetre.config(menu=menubar)
############################
menufichier = Menu(menubar,tearoff=0) 
menubar.add_cascade(label="Fichier", menu=menufichier) 
menufichier.add_command(label="Ouvrir ", command=ouvrir )

menufichier.add_separator() 
menufichier.add_command(label="Enregistrer") 
menufichier.add_separator() 
menufichier.add_command(label="Enregistrer sous", command= save )
menufichier.add_separator() 
menufichier.add_command(label="Quitter", command = _quit)

########################################
###############################""
menuBruit = Menu(menubar,tearoff=0) 
menubar.add_cascade(label="Bruit", menu=menuBruit) 
menuBruit.add_command(label="Poiver et sel ", command =poivreAndSelNoise) 
menuBruit.add_command(label="Gaussien", command =gaussianNoise)
##########################""""""""""


menuTransformation = Menu(menubar,tearoff=0) 
menubar.add_cascade(label="Transformation élémentaire", menu=menuTransformation) 
menuTransformation.add_command(label="Niveau de grais", command =grayScaleImage )
menuTransformation.add_command(label="Inverse", command =colorInversion )
menuTransformation.add_command(label="seuillage " , command =Seuillage3d )
menuTransformation.add_command(label="Contrast", command =contrast)
menuTransformation.add_command(label="Histogramme", command =hist)

################################
#################################""
menuFiltre1 = Menu(menubar,tearoff=0)

menubar.add_cascade(label="Filtre pass bas", menu=menuFiltre1) 

menuFiltre1.add_command(label="gaussien ", command = FGaussien)
menuFiltre1 .add_command(label="Moyenneur", command =ImageMoyenne) 
menuFiltre1 .add_command(label="Pyramidal", command =coniqueFilter) 
menuFiltre1 .add_command(label="Conique", command =pyramidalFilter) 

menuFiltre1 .add_command(label="Median", command = medianFilter) 
##########################################
####################################
menuFiltre2 = Menu(menubar,tearoff=0) 
menubar.add_cascade(label="Filtre pass haut", menu=menuFiltre2) 
menuFiltre2.add_command(label="Sobel", command =contourDetectionSobel) 
menuFiltre2.add_command(label="Prewitt", command =contourDetectionPrewitt) 
menuFiltre2.add_command(label="Roberts", command =contourDetectionRoberts) 

menuFiltre2.add_command(label="Gradient", command =contourDetectionGradient) 
menuFiltre2.add_command(label="Laplacian", command =contourDetectionLaplacian_gaussian)
################################
menufrequentiel = Menu(menubar,tearoff=0) 
menubar.add_cascade(label="Filtre frequentiel", menu=menufrequentiel) 
menufrequentiel.add_command(label="Filtre passe bas", command =imageEnhancementFFT) 
menufrequentiel.add_command(label="Filtre passe haut", command =contourDetectionFFT) 
menufrequentiel.add_command(label="Filtre passe bas de butterworth", command =ButterworthLowPass) 
menufrequentiel.add_command(label="Filtre passe haut de butterworth", command =contourDetectionButterworth) 
menufrequentiel.add_command(label="Filtre passe bonde", command =imageEnhancementFFTBandPass) 
####################################
Morphologie = Menu(menubar,tearoff=0) 
menubar.add_cascade(label="Morphologie mathématiques", menu=Morphologie) 
Morphologie.add_command(label="Erosion",command=Erosion)
Morphologie.add_command(label="Dilatation",command=Delation) 
Morphologie.add_command(label="Ouverture",command=ouverture)
Morphologie.add_command(label="Fermeture",command=fermeture)
Morphologie.add_command(label="chapeau haut de forme blanc",command=whiteTopHat)
Morphologie.add_command(label="chapeau haut de forme noir",command=blackTopHat)
Morphologie.add_command(label="Gradiant Morphologie",command=gradientM)

detection = Menu(menubar,tearoff=0) 
menubar.add_cascade(label="Point d'intérét", menu=detection ) 
detection.add_command(label="Susan",command=Susan)
detection.add_command(label="Harris",command=harris)

fenetre.config(menu=menubar)
fenetre.mainloop()
