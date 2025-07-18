import tkinter as tk
from PIL import Image, ImageTk
from subprocess import call

# Main Window Setup
root = tk.Tk()
root.title("Multimedia Deepfake Detection System")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry(f"{w}x{h}+0+0")
root.configure(bg="#f0f2f5")

# Background Image
bg_image = Image.open('neural_network_bg.jpg')
bg_image = bg_image.resize((w, h), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Title
title_label = tk.Label(root, text="Multimedia Deepfake Detection System", 
                       font=("Helvetica", 36, "bold"), 
                       bg="black", fg="white", pady=20)
title_label.pack(fill="x")

# Welcome Frame
welcome_frame = tk.LabelFrame(root, text="  WELCOME  ", 
                              font=("Helvetica", 16, "bold"), 
                              bg="grey", fg="#152238", bd=6, relief="ridge")
welcome_frame.pack(pady=80)
welcome_frame.pack_propagate(False)  # Allow inner widgets to size frame
welcome_frame.configure(width=400, height=300)

# Functions
def login():
    call(["python", "login.py"])

def reg():
    call(["python", "registration.py"])

def window():
    root.destroy()

# Button Style
button_font = ("Helvetica", 14, "bold")

def on_enter(e):
    e.widget['bg'] = "#dcdcdc"

def on_leave(e):
    e.widget['bg'] = "white"

# Button Factory
def create_button(parent, text, command):
    btn = tk.Button(parent, text=text, command=command,
                    font=button_font, bg="white", fg="black",
                    width=20, height=2, bd=2, relief="ridge",
                    activebackground="#e0e0e0", activeforeground="black")
    btn.pack(pady=10)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn

# Buttons
create_button(welcome_frame, "Login Now", login)
create_button(welcome_frame, "Register", reg)
create_button(welcome_frame, "Exit", window)

# Footer
footer = tk.Label(root, text="© 2025 Deepfake Detection | Secure Media Verification", 
                  font=("Helvetica", 10), bg="black", fg="white")
footer.pack(side="bottom", fill="x")

root.mainloop()
