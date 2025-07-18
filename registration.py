import customtkinter as ctk
from tkinter import messagebox
import sqlite3
import os
import sys
import random
from PIL import Image, ImageTk

# ===== SETTINGS =====
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ===== MAIN WINDOW =====
root = ctk.CTk()
root.title("Multimedia Deepfake Detection System Registration")
root.geometry("1280x850")
root.state("zoomed")

# ===== SOLID BACKGROUND =====
bg_frame = ctk.CTkFrame(root, fg_color="#0a0e17")
bg_frame.place(x=0, y=0, relwidth=1, relheight=1)

# Add circuit pattern overlay
for i in range(0, 1280, 40):
    ctk.CTkFrame(bg_frame, width=1, height=850, fg_color="#161b22").place(x=i, y=0)
for i in range(0, 850, 40):
    ctk.CTkFrame(bg_frame, width=1280, height=1, fg_color="#161b22").place(x=0, y=i)

# ===== GLASS PANEL (using solid colors) =====
container = ctk.CTkFrame(
    bg_frame,
    width=700,
    height=760,
    corner_radius=24,
    fg_color="#0d1117",  # Dark background
    border_width=1,
    border_color="#58a6ff"  # Light blue border
)
container.place(relx=0.5, rely=0.5, anchor="center")

# ===== HEADER =====
header = ctk.CTkFrame(container, fg_color="transparent")
header.pack(pady=(20, 10))

ctk.CTkLabel(
    header,
    text="🛡️",
    font=("Segoe UI", 48),
    text_color="#58a6ff"
).pack()

ctk.CTkLabel(
    header,
    text=" REGISTRATION",
    font=("Segoe UI Black", 24, "bold"),
    text_color="#f0f6fc"
).pack(pady=(5, 0))

ctk.CTkLabel(
    header,
    text="Multimedia Deepfake Detection System | Security Clearance Required",
    font=("Segoe UI", 11),
    text_color="#8b949e"
).pack()

# ===== FORM FIELDS =====
form_frame = ctk.CTkFrame(container, fg_color="transparent")
form_frame.pack(pady=10)

def create_field(label, var, show=""):
    field_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
    field_frame.pack(fill="x", padx=60, pady=(8, 0))
    
    ctk.CTkLabel(
        field_frame,
        text=label,
        font=("Segoe UI", 12, "bold"),
        text_color="#58a6ff",
        anchor="w"
    ).pack(side="left", padx=(0, 10))
    
    ctk.CTkEntry(
        field_frame,
        textvariable=var,
        placeholder_text=f"Enter {label}",
        show=show,
        width=400,
        height=40,
        corner_radius=8,
        fg_color="#161b22",
        border_color="#30363d",
        font=("Segoe UI", 12)
    ).pack(side="right")

# Create variables and fields
fullname = ctk.StringVar()
address = ctk.StringVar()
username = ctk.StringVar()
email = ctk.StringVar()
phoneno = ctk.StringVar()
age = ctk.StringVar()
password = ctk.StringVar()
password1 = ctk.StringVar()

fields = [
    ("Full Name", fullname),
    ("Address", address),
    ("Username", username),
    ("Email", email),
    ("Phone", phoneno),
    ("Age", age),
    ("Password", password, "*"),
    ("Confirm Password", password1, "*")
]

for field in fields:
    create_field(*field)

# ===== REGISTRATION FUNCTION =====
def register():
    if not all([fullname.get(), username.get(), password.get()]):
        messagebox.showerror("Error", "Required fields are missing!")
        return
        
    if password.get() != password1.get():
        messagebox.showerror("Error", "Passwords do not match!")
        return
    
    try:
        with sqlite3.connect('evaluation.db') as db:
            c = db.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS admin_registration (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fullname TEXT, address TEXT, username TEXT UNIQUE,
                        email TEXT, phoneno TEXT, age INTEGER, password TEXT)''')
            c.execute("INSERT INTO admin_registration VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)",
                     (fullname.get(), address.get(), username.get(), 
                      email.get(), phoneno.get(), age.get(), password.get()))
            db.commit()
            messagebox.showinfo("Success", "Registration successful!")
            root.destroy()
            os.system(f'"{sys.executable}" login.py')
    except sqlite3.IntegrityError:
        messagebox.showerror("Error", "Username already exists!")
    except Exception as e:
        messagebox.showerror("Error", f"Database error: {str(e)}")

# ===== BUTTONS =====
button_frame = ctk.CTkFrame(container, fg_color="transparent")
button_frame.pack(pady=(20, 10))

register_btn = ctk.CTkButton(
    button_frame,
    text="COMPLETE REGISTRATION",
    command=register,
    width=550,
    height=45,
    fg_color="#238636",
    hover_color="#2ea043",
    font=("Segoe UI Semibold", 14),
    corner_radius=10
)
register_btn.pack(pady=5)

return_btn = ctk.CTkButton(
    button_frame,
    text="RETURN TO LOGIN",
    command=lambda: [root.destroy(), os.system(f'"{sys.executable}" login.py')],
    width=550,
    height=45,
    fg_color="transparent",
    hover_color="rgba(88, 166, 255, 0.1)",
    text_color="#58a6ff",
    font=("Segoe UI Semibold", 14),
    border_width=1,
    border_color="#58a6ff",
    corner_radius=10
)
return_btn.pack(pady=5)

# ===== FOOTER =====
ctk.CTkLabel(
    container,
    text="© 2025 Multimedia Deepfake Detection  System | v4.2.1 |  Encrypted",
    font=("Consolas", 10),
    text_color="#6e7681"
).pack(pady=(10, 20))

root.mainloop()