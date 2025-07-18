import tkinter as tk
from tkinter import ttk
from PIL import Image , ImageTk
import csv
from datetime import date
import time
import numpy as np
import cv2
from tkinter.filedialog import askopenfilename
import os
import shutil
import keras
from tensorflow import keras 
import Train_FDD_cnn as TrainM
from gtts import gTTS

#==============================================================================

root = tk.Tk()
root.state('zoomed')

root.title("Multimedia Deepfake Detection System")

current_path = str(os.path.dirname(os.path.realpath('__file__')))
basepath = current_path + "/"

#==============================================================================
#==============================================================================

img = Image.open(basepath + "f1.jpeg")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
bg = img.resize((w, h))
bg_img = ImageTk.PhotoImage(bg)
bg_lbl = tk.Label(root, image=bg_img)
bg_lbl.place(x=0, y=0)
# Create a frame to simulate a stylish header bar
header_frame = tk.Frame(root, bg="#0f172a", height=70, width=w)
header_frame.place(x=0, y=0)

# Add a stylish heading label
heading = tk.Label(
    header_frame,
    text="Multimedia Deepfake Detection System",
    font=("Segoe UI", 22, "bold"),
    bg="#0f172a",
    fg="#f1f5f9"
)
heading.place(relx=0.5, rely=0.5, anchor="center")
#============================================================================================================

def create_folder(FolderN):
    dst = os.getcwd() + "/" + FolderN
    if not os.path.exists(dst):
        os.makedirs(dst)
    else:
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)

def CLOSE():
    root.destroy()

def update_label(str_T):
    result_label = tk.Label(root, text=str_T, width=50, font=("bold", 25), bg='cyan', fg='black')
    result_label.place(x=400, y=400)

def train_model():
    update_label("Model Training Start...............")
    start = time.time()
    X = TrainM.main()
    end = time.time()
    ET = "Execution Time: {0:.4} seconds \n".format(end - start)
    msg = "Model Training Completed.." + '\n' + X + '\n' + ET
    update_label(msg)

def run_video(VPathName, XV, YV, S1, S2):
    cap = cv2.VideoCapture(VPathName)
    def show_frame():
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_FPS, 30)
        out = cv2.transpose(frame)
        out = cv2.flip(out, flipCode=0)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image).resize((S1, S2))
        imgtk = ImageTk.PhotoImage(image=img)
        lmain = tk.Label(root)
        lmain.place(x=XV, y=YV)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)
    show_frame()

def VIDEO():
    global fn
    fn = ""
    fileName = askopenfilename(initialdir='/dataset', title='Select image',
                               filetypes=[("all files", "*.*")])
    fn = fileName
    Sel_F = fileName.split('/').pop()
    Sel_F = Sel_F.split('.').pop(1)
    if Sel_F != 'mp4':
        print("Select Video .mp4 File!!!!!!")
    else:
        run_video(fn, 560, 190, 753, 485)

def show_FDD_video(video_path):
    from keras.models import load_model
    import cv2
    import numpy as np

    img_cols, img_rows = 64, 64
    FALLModel = load_model('model11.h5', compile=False)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("{} cannot be opened".format(video_path))

    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)
    red = (0, 0, 255)
    line_type = cv2.LINE_AA
    i = 1

    while True:
        ret, frame = video.read()
        if not ret:
            break

        img = cv2.resize(frame, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img)
        X_img = img.reshape(-1, img_cols, img_rows, 1)
        X_img = X_img.astype('float32') / 255.0

        predicted = FALLModel.predict(X_img)
        label = 1 if predicted[0][0] < 0.5 else 0

        frame_num = int(i)
        color = red if label == 1 else green
        label_text = "Fake Video Detected" if label == 1 else "Original Video Detected"

        print(label_text)

        # Draw thin labels
        font_scale = 0.6  # Smaller size
        thickness = 1     # Thinner text

        frame = cv2.putText(frame, f"Frame: {frame_num}", (5, 25), font, font_scale, color, thickness, line_type)
        frame = cv2.putText(frame, f"Label: {label_text}", (5, 50), font, font_scale, color, thickness, line_type)

        i += 1
        cv2.imshow('FDD', frame)
        if cv2.waitKey(30) == 27:  # ESC key
            break

    video.release()
    cv2.destroyAllWindows()

def Video_Verify():
    global fn
    fileName = askopenfilename(
        initialdir='E:/Softech mayuri code/Mayuri Groups/Mayuri Groups/23SS312-Deepfake/deepfake/Video/c1.mp4',
        title='Select image',
        filetypes=[("all files", "*.*")]
    )
    fn = fileName
    Sel_F = fileName.split('/').pop()
    Sel_F = Sel_F.split('.').pop(1)
    if Sel_F != 'mp4':
        print("Select Video File!!!!!!")
    else:
        show_FDD_video(fn)

# ================== Modernized Buttons ==================

# REMOVE if no title require
# heading = tk.Label(
#     root,
#     text="Deep Fake Detection",
#     width=120,
#     height=2,
#     font=("Times New Roman", 24, "bold"),  # Title font
#     bg="white",  # Background of title
#     fg="black"   # Text color
# )
# heading.place(x=0, y=0)


# Update the button style:

button_style = {
    "font": ("Helvetica", 14, "bold"),
    "width": 25,
    "height": 2,
    "bg": "white",  # White background
    "fg": "black",  # Bold black text
    "activebackground": "#f0f0f0",
    "activeforeground": "black",
    "bd": 1,
    "highlightthickness": 1,
    "highlightbackground": "#cccccc",
    "cursor": "hand2"
}

# Apply to your buttons:

button5 = tk.Button(root, command=Video_Verify, text="Select Video", **button_style)
button5.place(x=100, y=200)

button6 = tk.Button(root, command=train_model, text="Video Train", **button_style)
button6.place(x=100, y=270)

close = tk.Button(root, command=CLOSE, text="Exit", **button_style)
close.place(x=100, y=340)


# ========================================================

root.mainloop()
