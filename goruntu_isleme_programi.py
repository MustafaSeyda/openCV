import tkinter as tk
from skimage import data
from skimage.util import img_as_ubyte
from skimage.morphology import (disk, erosion, dilation, opening, closing, white_tophat,
                                black_tophat, skeletonize, convex_hull_image, flood_fill, remove_small_holes)
import cv2
import numpy as np
from matplotlib import pyplot as plt

window=tk.Tk()
window.geometry("1600x300+100+50")


def boyutlandırma(img=str,genişlik=int,uzunluk=int):
    image=cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image=cv2.resize(image,(genişlik,uzunluk))
    cv2.imwrite("resized_image.png",resized_image)
    plt.imshow(resized_image)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


label1=tk.Label(text="Görüntünün İsmi")
label1.place(x=10,y=35)
entry1=tk.Entry()
entry1.place(x=10,y=55,width=100)
label2=tk.Label(text="Genişlik")
label2.place(x=10,y=75)
entry2=tk.Entry()
entry2.place(x=10,y=95,width=100)
label3=tk.Label(text="Uzunluk")
label3.place(x=10,y=115)
entry3=tk.Entry()
entry3.place(x=10,y=135,width=100)


def boyutlandırma_entry():
    a=str(entry1.get())
    b=int(entry2.get())
    c=int(entry3.get())
    boyutlandırma(a,b,c)


buton1=tk.Button(window,text="boyutlandırma",command=boyutlandırma_entry)
buton1.place(x=10,y=5,width=100)




def taşıma(img=str,tx=int,ty=int):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rows,cols=image.shape[:2]
    translation_matrix=np.float32([[1,0,tx],[0,1,ty]])
    image_translation=cv2.warpAffine(image,translation_matrix,(cols,rows))
    cv2.imwrite("image_translation.png",image_translation)
    show_comparison(image, image_translation, "taşıma")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


label4=tk.Label(text="Görüntünün İsmi")
label4.place(x=150,y=35)
entry4=tk.Entry()
entry4.place(x=150,y=55,width=100)
label5=tk.Label(text="X Uzunluğu")
label5.place(x=150,y=75)
entry5=tk.Entry()
entry5.place(x=150,y=95,width=100)
label6=tk.Label(text="Y Uzunluğu")
label6.place(x=150,y=115)
entry6=tk.Entry()
entry6.place(x=150,y=135,width=100)


def taşıma_entry():
    a=str(entry4.get())
    b=int(entry5.get())
    c=int(entry6.get())
    taşıma(a,b,c)


buton2=tk.Button(window,text="taşıma",command=taşıma_entry)
buton2.place(x=150,y=5,width=100)



def döndürme(img=str,açı=float,oran=float):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rows, cols = image.shape[:2]
    rotation_matrix=cv2.getRotationMatrix2D((cols/2,rows/2),açı,oran)
    image_rotation=cv2.warpAffine(image,rotation_matrix,(cols,rows))
    cv2.imwrite("image_rotation.png", image_rotation)
    show_comparison(image, image_rotation, "döndürme")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

label7=tk.Label(text="Görüntünün İsmi")
label7.place(x=290,y=35)
entry7=tk.Entry()
entry7.place(x=290,y=55,width=100)
label8=tk.Label(text="Açı")
label8.place(x=290,y=75)
entry8=tk.Entry()
entry8.place(x=290,y=95,width=100)
label9=tk.Label(text="Boyut Değişim Oranı")
label9.place(x=290,y=115)
entry9=tk.Entry()
entry9.place(x=290,y=135,width=100)


def döndürme_entry():
    a=str(entry7.get())
    b=int(entry8.get())
    c=float(entry9.get())
    döndürme(a,b,c)


buton3=tk.Button(window,text="döndürme",command=döndürme_entry)
buton3.place(x=290,y=5,width=100)


def eşikleme(img=str,low=int,high=int,type=str):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    type=type.upper()
    if type=="BINARY":
        e,thresh=cv2.threshold(image,low,high,cv2.THRESH_BINARY)
        cv2.imwrite("BINARY.png",thresh)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
        show_comparison(image, thresh, "BINARY")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if type == "BINARY INV":
        e, thresh = cv2.threshold(image, low, high, cv2.THRESH_BINARY_INV)
        cv2.imwrite("BINARY_INV.png", thresh)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
        show_comparison(image, thresh, "BINARY INV")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if type=="OTSU":
        e,thresh=cv2.threshold(image,low,high,cv2.THRESH_OTSU)
        cv2.imwrite("OTSU.png",thresh)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
        show_comparison(image, thresh, "OTSU")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if type=="TOZERO":
        e,thresh=cv2.threshold(image,low,high,cv2.THRESH_TOZERO)
        cv2.imwrite("TOZERO.png",thresh)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
        show_comparison(image, thresh, "TOZERO")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if type=="TOZERO INV":
        e,thresh=cv2.threshold(image,low,high,cv2.THRESH_TOZERO_INV)
        cv2.imwrite("TOZERO_INV.png",thresh)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
        show_comparison(image, thresh, "TOZERO INV")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if type=="TRUNC":
        e,thresh=cv2.threshold(image,low,high,cv2.THRESH_TRUNC)
        cv2.imwrite("TRUNC.png",thresh)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
        show_comparison(image,thresh,"TRUNCH")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


label10=tk.Label(text="Görüntünün İsmi")
label10.place(x=430,y=35)
entry10=tk.Entry()
entry10.place(x=430,y=55,width=100)
label11=tk.Label(text="Alt Eşik")
label11.place(x=430,y=75)
entry11=tk.Entry()
entry11.place(x=430,y=95,width=100)
label12=tk.Label(text="Üst Eşik")
label12.place(x=430,y=115)
entry12=tk.Entry()
entry12.place(x=430,y=135,width=100)
label13=tk.Label(text="Eşikleme Türü")
label13.place(x=430,y=155)
entry13 = tk.Entry()
entry13.place(x=430, y=175, width=100)


def eşikleme_entry():
    a=str(entry10.get())
    b=int(entry11.get())
    c=int(entry12.get())
    d=str(entry13.get())
    eşikleme(a,b,c,d)


buton4=tk.Button(window,text="eşikleme",command=eşikleme_entry)
buton4.place(x=430,y=5,width=100)


def projective_transformation(img=str,ax=int,ay=int,bx=int,by=int,cx=int,cy=int,dx=int,dy=int):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rows, cols = image.shape[:2]
    src_points = np.float32([
        [ax,ay],
        [bx,by],
        [cx,cy],
        [dx,dy]])
    dst_points = np.float32([
        [0, 0],
        [cols-1,0],
        [0,rows-1],
        [cols-1,rows-1]])
    projective_matrix=cv2.getPerspectiveTransform(src_points,dst_points)
    image_output=cv2.warpPerspective(image,projective_matrix,(cols,rows))
    cv2.imwrite("projective_img_output.png", image_output)
    show_comparison(image,image_output,"projective transformation")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

label14=tk.Label(text="Görüntünün İsmi")
label14.place(x=641,y=35)
entry14=tk.Entry()
entry14.place(x=641,y=55,width=150)
label15=tk.Label(text="1. Nokta x noktası")
label15.place(x=570,y=75)
entry15=tk.Entry()
entry15.place(x=570,y=95,width=150)
label16=tk.Label(text="1. Nokta y noktası")
label16.place(x=722,y=75)
entry16=tk.Entry()
entry16.place(x=722,y=95,width=150)
label17=tk.Label(text="2. Nokta x noktası")
label17.place(x=570,y=115)
entry17 = tk.Entry()
entry17.place(x=570, y=135, width=150)
label18=tk.Label(text="2. Nokta y noktası")
label18.place(x=722,y=115)
entry18 = tk.Entry()
entry18.place(x=722, y=135, width=150)
label19=tk.Label(text="3. Nokta x noktası")
label19.place(x=570,y=155)
entry19=tk.Entry()
entry19.place(x=570,y=175,width=150)
label20=tk.Label(text="3. Nokta y noktası")
label20.place(x=722,y=155)
entry20=tk.Entry()
entry20.place(x=722,y=175,width=150)
label21=tk.Label(text="4. Nokta x noktası")
label21.place(x=570,y=195)
entry21 = tk.Entry()
entry21.place(x=570, y=215, width=150)
label22=tk.Label(text="4. Nokta y noktası")
label22.place(x=722,y=195)
entry22 = tk.Entry()
entry22.place(x=722, y=215, width=150)


def projective_transformation_entry():
    a=str(entry14.get())
    b=int(entry15.get())
    c=int(entry16.get())
    d=int(entry17.get())
    e=int(entry18.get())
    b2 =int(entry19.get())
    c2 = int(entry20.get())
    d2 = int(entry21.get())
    e2 = int(entry22.get())
    projective_transformation(a,b,b2,c,c2,d,d2,e,e2)


buton5=tk.Button(window,text="projective_transformation",command=projective_transformation_entry)
buton5.place(x=641,y=5,width=200)


def affine_transformation(img=str):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rows,cols=image.shape[:2]
    src_points=np.float32([
        [0,0],
        [cols-1,0],
        [0,rows-1]])
    dst_points=np.float32([
        [0,0],
        [int(0.6*(cols-1)),0],
        [int(0.4*(cols-1)),rows-1]])
    affine_matrix=cv2.getAffineTransform(src_points,dst_points)
    img2_output=cv2.warpAffine(image,affine_matrix,(cols,rows))
    cv2.imwrite("affine_img_output.png",img2_output)
    show_comparison(image, img2_output, "affine transformation")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def affine_transformation_entry():
    a = str(entry23.get())
    affine_transformation(a)


label23=tk.Label(text="Görüntünün İsmi")
label23.place(x=912,y=35)
entry23 = tk.Entry()
entry23.place(x=912, y=55, width=150)
buton6=tk.Button(window,text="affine_transformation",command=affine_transformation_entry)
buton6.place(x=912,y=5,width=180)


def foto_historgram(image,L):
    histogram,bins =np.histogram(image,bins=L,range=(0,L))
    return histogram


def normalleştirilmiş_histogram(image,L):
    histogram=foto_historgram(image,L)
    return histogram/image.size


def kumulatif_dağılım_oluştur(p_r_r):
    return np.cumsum(p_r_r)


def histogram_eşitleme(image,L):
    p_r_r=normalleştirilmiş_histogram(image,L)
    kumulatif_dağılım=kumulatif_dağılım_oluştur(p_r_r)
    dönüşüm_fonksiyonu=(L-1)*kumulatif_dağılım
    shape=image.shape
    ravel=image.ravel()
    hist_eş_foto=np.zeros_like(ravel)
    for i, pixel in enumerate(ravel):
        hist_eş_foto[i]=dönüşüm_fonksiyonu[pixel]
    return (hist_eş_foto.reshape(shape).astype(np.uint8))
def histogram(açık=str,koyu=str,gri=str):
    L=2**8
    açık2=cv2.imread(açık,0)
    koyu2=cv2.imread(koyu,0)
    gri2=cv2.imread(gri,0)
    hist_eş_açık2=histogram_eşitleme(açık2,L)
    hist_eş_koyu2=histogram_eşitleme(koyu2,L)
    hist_eş_gri2=histogram_eşitleme(gri2,L)
    yanyana_açık=np.hstack((açık2,hist_eş_açık2))
    yanyana_koyu = np.hstack((koyu2, hist_eş_koyu2))
    yanyana_gri = np.hstack((gri2, hist_eş_gri2))
    grid=np.vstack((yanyana_açık,yanyana_koyu,yanyana_gri))
    plt.imshow(grid,cmap="gray")
    plt.show()

label24=tk.Label(text="Açık Renkli Görüntü İsmi")
label24.place(x=1102,y=35)
entry24 = tk.Entry()
entry24.place(x=1102, y=55, width=150)
label25=tk.Label(text="Koyu Renkli Görüntü İsmi")
label25.place(x=1102,y=75)
entry25 = tk.Entry()
entry25.place(x=1102, y=95, width=150)
label26=tk.Label(text="Gri Renkli Görüntü İsmi")
label26.place(x=1102,y=115)
entry26 = tk.Entry()
entry26.place(x=1102, y=135, width=150)

def histogram_entry():
    a = str(entry24.get())
    b = str(entry25.get())
    c = str(entry26.get())
    histogram(a,b,c)

buton7=tk.Button(window,text="histogram",command=histogram_entry)
buton7.place(x=1102,y=5,width=150)


def rescale_intensity(img, upper, lower) :

    normalizedImg = np.zeros((800, 800))

    resc_inten = cv2.normalize(img,  normalizedImg, int(lower), int(upper), cv2.NORM_MINMAX)

    cv2.imshow('Rescale Intensity', resc_inten)

    cv2.waitKey(0)

    cv2.destroyWindow('Rescale Intensity')

    channel_initials = list('BGR')

    for channel_index in range(3):

        channel = np.zeros(shape=img.shape, dtype=np.uint8)

        channel[:,:,channel_index] = img[:,:,channel_index]

        cv2.imshow(f'{channel_initials[channel_index]}-RGB', channel)

    cv2.waitKey(0)


def filtreler():
    img = cv2.imread("images.jpg",0)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    kernel = np.ones((5 , 5), np.float32) / 25

    dst = cv2.filter2D(img, -1, kernel)

    blur = cv2.blur(img, (5,5))

    gblur = cv2.GaussianBlur(img, (5,5), 0)

    median = cv2.medianBlur(img, 5)

    laplacian = cv2.Laplacian(img , -1)

    boxFilter = cv2.boxFilter(img, 0, (7,7),cv2.BORDER_DEFAULT)

    bilateralFilter = cv2.bilateralFilter(img, 9, 5, 5)

    edges = cv2.Canny(img,100,100)


    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th2 = cv2.adaptiveThreshold(img_grey,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)


    titles = ['image', '2D Convolution',  'blur', 'GaussianBlur', 'median', 'bilateralFilter', 'edges' ,
    'Laplacian', 'Box Filter', 'Global Thrs', 'Adp Mean Thrs', 'Adp Gaus Thrs']
    images = [img, dst,  blur, gblur, median, bilateralFilter, edges, laplacian, boxFilter, th1, th2, th3]

    for i in range(12):
        plt.subplot(3, 4, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()


# ---------- 10 Adet Morfolojik İşlem Örneği ----------

footprint = disk(6)

ornek_gorsel = img_as_ubyte(data.shepp_logan_phantom())
ornek_gorsel_2 = data.horse()
ornek_gorsel_3 = data.binary_blobs()


# yeni pencerede orijinali ile işlem görmüş görsel karşılaştırması
def show_comparison(original, changed, title):
    titles = ['original', title]
    images = [original, changed]

    for i in range(2):
        plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


# 1) Erosion
def erosion_function(original):
    return erosion(original, footprint)


# 2) Dilation
def dilation_function(original):
    return dilation(original, footprint)


# 3) Opening
def opening_function(original):
    return opening(original, footprint)


# 4) Closing
def closing_function(original):
    return closing(original, footprint)


# 5) White tophat
def white_tophat_function(original):
    return white_tophat(original, footprint)


# 6) Black tophat
def black_tophat_function(original):
    return black_tophat(original, footprint)


# 7) Skeletonize
def skeletonize_function(original):
    return skeletonize(original == 0)


# 8) Convex hull
def convex_hull_function(original):
    return convex_hull_image(original == 0)


# 9) Flood Fill
def flood_fill_function(original):
    return flood_fill(original, (1, 1), 1000)


# 10) Removing Small Holes
def remove_small_holes_function(original):
    return remove_small_holes(original, 100000)


# 10 adet morfolojik işlem örneği bu fonksiyon ile çağırabilir.
# argüman olarak 1'den 10'a kadar sayılar girilmesi yeterlidir.
def morphology_10(method):
    if method == 1:
        show_comparison(ornek_gorsel, erosion_function(ornek_gorsel), 'eroded')
    elif method == 2:
        show_comparison(ornek_gorsel, dilation_function(ornek_gorsel), 'dilated')
    elif method == 3:
        show_comparison(ornek_gorsel, opening_function(ornek_gorsel), 'opened')
    elif method == 4:
        phantom = ornek_gorsel.copy()
        phantom[10:30, 200:210] = 0
        show_comparison(phantom, closing_function(phantom), 'closed')
    elif method == 5:
        phantom = ornek_gorsel.copy()
        phantom[340:350, 200:210] = 255
        phantom[100:110, 200:210] = 0
        show_comparison(phantom, white_tophat_function(phantom), 'white top hatted')
    elif method == 6:
        phantom = ornek_gorsel.copy()
        phantom[340:350, 200:210] = 255
        phantom[100:110, 200:210] = 0
        show_comparison(phantom, black_tophat_function(phantom), 'black top hatted')
    elif method == 7:
        show_comparison(ornek_gorsel_2, skeletonize_function(ornek_gorsel_2), 'skeletonized')
    elif method == 8:
        show_comparison(ornek_gorsel_2, convex_hull_function(ornek_gorsel_2), 'convex_hulled')
    elif method == 9:
        show_comparison(ornek_gorsel, flood_fill_function(ornek_gorsel), 'flood filled')
    elif method == 10:
        show_comparison(ornek_gorsel_3, remove_small_holes_function(ornek_gorsel_3), 'small holes removed')
    else:
        pencere2 = tk.Tk()
        label2 = tk.Label(pencere2, text=("hatalı giriş yaptınız. 1 ile 10 arasında bir sayı giriniz. "))
        label2.pack()


label27=tk.Label(text="1-10 Arasında Bir Sayı Girin")
label27.place(x=1292,y=35)
entry27 = tk.Entry()
entry27.place(x=1292, y=55, width=150)


def morfoloji_entry():
    a = int(entry27.get())
    morphology_10(a)


buton8=tk.Button(window,text="morfoloji",command=morfoloji_entry)
buton8.place(x=1292,y=5,width=150)


def kamera():

 #bilgisayara bagli ve aktif olan kameralardan 0.yı secer.
    camera = cv2.VideoCapture(0)

    while True:
        _, frame = camera.read()

        cv2.imshow("Camera", frame)

        kenar_belirleme = cv2.Canny(frame, 107, 119)
        cv2.imshow('Kenar Belirleme', kenar_belirleme)

    #programi kapatmak icin x'e basiniz.
        if cv2.waitKey(5) == ord('x'):
            break

    camera.release()
    cv2.destroyAllWindows()


buton9=tk.Button(window,text="video işleme",command=kamera)
buton9.place(x=1292,y=85,width=150)

label27=tk.Label(text="Kamerayı kapatmak için x'e basın ")
label27.place(x=1290,y=110)


window.mainloop()
