import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk 
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import CNNModel 
import sqlite3
#import tfModel_test as tf_test
global fn
fn=""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="#152238")  # Dark blue background
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Multimedia Deepfake Detection System")


#430
#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 = Image.open('neural_network_bg.jpg')
image2 = image2.resize((w, h), Image.LANCZOS)  # Full screen fit

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)
background_label.image = background_image
background_label.place(x=0, y=0, relwidth=1, relheight=1)  # Full coverage
#
lbl = tk.Label(root, 
               text="Multimedia Deepfake Detection System", 
               font=('Helvetica', 35, 'bold'), 
               width=65, height=1,
               bg="#152238", fg="white")
lbl.place(x=0, y=0)


frame_alpr = tk.LabelFrame(root, 
                          text=" Image Processing ", 
                          width=920, height=120, 
                          bd=5, 
                          font=('Helvetica', 14, 'bold'),
                          bg="#2c3e50", fg="white",
                          relief=tk.RIDGE)
frame_alpr.place(relx=0.5, y=650, anchor='center')  # Centered


def update_label1(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=300, y=650)
    
    
    
################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
def update_cal(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=60, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=290, y=350)
    
    
    
###########################################################################
def train_model():
 
    update_label("Model Training Start...............")
    
    start = time.time()

    X= CNNModel.main()
    print(X)
    
    end = time.time()
        
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
    msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

    update_label(msg)

import functools
import operator


def convert_str_to_tuple(tup):
    s = functools.reduce(operator.add, (tup))
    return s

def test_model_proc(fn):
    from keras.models import load_model


    
    
    IMAGE_SIZE = 64
    LEARN_RATE = 1.0e-4
    CH=3
    print(fn)
    if fn!="":
        # Model Architecture and Compilation
       
        model = load_model('model.h5',compile=False)
            
        # adam = Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
        img = Image.open(fn)
        img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
        img = np.array(img)
        
        
        

        
        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)
        
        img = img.astype('float32')
        img = img / 255.0
        print('img shape:',img)
        prediction = model.predict(img)
        print(np.argmax(prediction))
        cell=np.argmax(prediction)
        print(cell)
        
        
                
        # myfile = open("1.txt")
        # txt = myfile.read()
        # print(txt)
        # myfile.close()
                
        
        
        if cell == 0:
            Cd="Real Image Detected."
            
        elif cell == 1:
            Cd="TAMPERED Image Detected."
            
        elif cell == 2:
            Cd="Fake Image Detected"
            
        
            
        A=Cd
        return A
    img = open(fn)    
    myfile = open("/Image Foregery Detection 100% Code/Image Foregery Detection 100% Code/testing")
    if img == myfile:
          print(myfile)


# def clear_img():
    
#     img11 = tk.Label(frame_display, background='lightblue4',width=160,height=120)
#     img11.place(x=0, y=0)

def update_label(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=60, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=290, y=420)
# def train_model():
    
#     update_label("Model Training Start...............")
    
#     start = time.time()

#     X=Model_frm.main()
    
#     end = time.time()
        
#     ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
#     msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

#     update_label(msg)

def test_model():
    global fn
    if fn!="":
        update_label("Model Testing Start...............")
        
        start = time.time()
    
        X=test_model_proc(fn)
        
        X1="Selected Image is {0}".format(X)
        
        end = time.time()
            
        ET="Execution Time: {0:.4} seconds \n".format(end-start)
        
        msg="Image Testing Completed.."+'\n'+ X1 + '\n'+ ET
        fn=""
    else:
        msg="Please Select Image For Prediction...."
        
    update_label(msg)
    
    
def openimage():
   
    global fn
    print(fn)
    fileName = askopenfilename(initialdir='C:/Users/Sagar/Desktop/Image Foregery Detection 100% Code/testing', title='Select image for Aanalysis ',filetypes=[("all files", "*.*")])
    IMAGE_SIZE=200
    imgpath = fileName
    fn = fileName


#        img = Image.open(imgpath).convert("L")
    img = Image.open(imgpath)
    
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
#        img = img / 255.0
#        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)


    x1 = int(img.shape[0])
    y1 = int(img.shape[1])


#
#        gs = cv2.cvtColor(cv2.imread(imgpath, 1), cv2.COLOR_RGB2GRAY)
#
#        gs = cv2.resize(gs, (x1, y1))
#
#        retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    img = tk.Label(root,text='Original',font=('times new roman', 20 ,'bold'), image=imgtk,compound='bottom', height=250, width=250)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250)
    #result_label1.place(x=300, y=100)
    img.image = imgtk
    img.place(x=300, y=100)
   # out_label.config(text=imgpath)

def convert_grey():
    global fn 
    print(fn)
    IMAGE_SIZE=200
    
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
    
    x1 = int(img.shape[0])
    y1 = int(img.shape[1])

    gs = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_RGB2GRAY)

    gs = cv2.resize(gs, (x1, y1))

    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(threshold)

    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250, font=("bold", 25), bg='bisque2', fg='black',height=250)
    #result_label1.place(x=300, y=400)
    img2 = tk.Label(root,text='Gray',font=('times new roman', 20 ,'bold'), image=imgtk,compound='bottom', height=250, width=250,bg='white')
    img2.image = imgtk
    img2.place(x=580, y=100)
    #label_l1 = tk.Label(root, text='Gray' ,compound='bottom', width=4, height=1)
    #label_l1.place(x=690, y=110)

    im = Image.fromarray(threshold)
    imgtk = ImageTk.PhotoImage(image=im)

    img3 = tk.Label(root,text='Binary',font=('times new roman', 20 ,'bold'), image=imgtk,compound='bottom', height=250, width=250)
    img3.image = imgtk
    img3.place(x=880, y=100)
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250, font=("bold", 25), bg='bisque2', fg='black')
    #result_label1.place(x=300, y=400)
    
    
    # def percentage(Width, Depth):
    #   return 100 * float(Width)/float(Depth)

    # print(percentage(5, 7))
    # # If you want to limit the number of decimal to 2, change the number in {:.2f} as you wish;
    # print('{:.2f}'.format(percentage(5, 7)))
    # your_value = 1/3.0
    # print('{:.1%}'.format(your_value)) # Change the "1" to however many decimal places you need
    # # Result:
    # # '33.3%'


#################################################################################################################
def window():
    from subprocess import call
    call(["python", "GUI_Master_New.py"])   

    # root.destroy()
    




button1 = tk.Button(frame_alpr, text=" Select_Image ", command=openimage,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button1.place(x=100, y=30)

button2 = tk.Button(frame_alpr, text="Image_preprocess", command=convert_grey, width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button2.place(x=300, y=30)

# button3 = tk.Button(frame_alpr, text="Train Model", command=train_model, width=12, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
# button3.place(x=10, y=160)

button4 = tk.Button(frame_alpr, text="CNN_Prediction", command=test_model,width=15, height=1,bg="white",fg="black", font=('times', 15, ' bold '))
button4.place(x=500, y=30)

#button5 = tk.Button(frame_alpr, text="button5", command=window,width=8, height=1, font=('times', 15, ' bold '),bg="yellow4",fg="white")
#button5.place(x=450, y=20)



exit = tk.Button(frame_alpr, 
                text="Next", 
                command=window,
                width=15,
                height=1,
                font=('times', 15, 'bold'),
                bg="white",      # White background
                fg="black",      # Black text
                activebackground="#f0f0f0",  # Slightly darker white on hover
                activeforeground="black",    # Keep text black on hover
                relief=tk.RAISED,  # 3D raised effect
                bd=2)             # Border width
exit.place(x=700, y=30)




root.mainloop()


