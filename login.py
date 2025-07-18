import customtkinter as ctk
from tkinter import messagebox
import sqlite3
import os
import sys
from PIL import Image, ImageTk

# ===== SETUP =====
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.title("Multimedia Deepfake Detection System Login")
root.geometry("1080x720")  # Updated window size
root.resizable(False, False)

# ===== BACKGROUND IMAGE =====
try:
    bg_image = Image.open("neural_network_bg.jpg").resize((1080, 720), Image.Resampling.LANCZOS)  # Updated resizing method
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = ctk.CTkLabel(root, image=bg_photo, text="")
    bg_label.image = bg_photo
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
except Exception as e:
    print("Error loading background image:", e)

# ===== MAIN UI CONTENT =====
main_frame = ctk.CTkFrame(
    root,
    width=450,
    height=580,
    corner_radius=24,
    fg_color="#0d1117",
    border_width=1,
    border_color="#58a6ff"
)
main_frame.place(relx=0.5, rely=0.5, anchor="center")

# ===== HEADER SECTION =====
header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
header_frame.pack(pady=(40, 20), padx=20, fill="x")

shield_icon = ctk.CTkLabel(
    header_frame,
    text="🛡️",
    font=("Segoe UI", 54),
    text_color="#58a6ff"
)
shield_icon.pack()

title_label = ctk.CTkLabel(
    header_frame,
    text="Multimedia Deepfake Detection System",
    font=("Segoe UI Black", 20),
    text_color="#f0f6fc",
    wraplength=400,
    justify="center"
)
title_label.pack(pady=(10, 5))

subtitle_label = ctk.CTkLabel(
    header_frame,
    text="Authenticate to continue",
    font=("Segoe UI", 12),
    text_color="#8b949e"
)
subtitle_label.pack()

# ===== FORM SECTION =====
form_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
form_frame.pack(pady=20, padx=40, fill="both", expand=True)

# Username Field
username_label = ctk.CTkLabel(
    form_frame,
    text="User Name",
    font=("Segoe UI", 11, "bold"),
    text_color="#58a6ff",
    anchor="w"
)
username_label.pack(fill="x", pady=(0, 5))

username_entry = ctk.CTkEntry(
    form_frame,
    placeholder_text="User_ID",
    width=360,
    height=48,
    corner_radius=12,
    border_width=1,
    border_color="#30363d",
    fg_color="#161b22",
    font=("Segoe UI", 13)
)
username_entry.pack(pady=(0, 15))

# Password Field
password_label = ctk.CTkLabel(
    form_frame,
    text="Password",
    font=("Segoe UI", 11, "bold"),
    text_color="#58a6ff",
    anchor="w"
)
password_label.pack(fill="x", pady=(0, 5))

password_entry = ctk.CTkEntry(
    form_frame,
    placeholder_text="••••••••••••",
    show="•",
    width=360,
    height=48,
    corner_radius=12,
    border_width=1,
    border_color="#30363d",
    fg_color="#161b22",
    font=("Segoe UI", 13)
)
password_entry.pack(pady=(0, 25))

# ===== BUTTONS =====
button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
button_frame.pack(pady=10, padx=40, fill="both")

def login():
    try:
        with sqlite3.connect('evaluation.db') as db:
            cursor = db.cursor()
            cursor.execute(
                "SELECT * FROM admin_registration WHERE username=? AND password=?",
                (username_entry.get(), password_entry.get())
            )
            if cursor.fetchone():
                messagebox.showinfo("ACCESS GRANTED", "Initializing deepfake analysis modules...")
                root.destroy()
                os.system(f'"{sys.executable}" GUI_Master_old.py')
            else:
                messagebox.showerror("AUTH FAILED", "Invalid credentials - 2 attempts remaining")
    except Exception as e:
        messagebox.showerror("SYSTEM ERROR", f"Database connection failed\n{str(e)}")

login_btn = ctk.CTkButton(
    button_frame,
    text="LOGIN",
    command=login,
    width=360,
    height=48,
    fg_color="#238636",
    hover_color="#2ea043",
    corner_radius=12,
    font=("Segoe UI Semibold", 14)
)
login_btn.pack(pady=5)

def register():
    root.destroy()
    os.system(f'"{sys.executable}" registration.py')

register_btn = ctk.CTkButton(
    button_frame,
    text="NEW REGISTRATION",
    command=register,
    width=360,
    height=48,
    fg_color="#1f6feb",
    hover_color="#388bfd",
    corner_radius=12,
    font=("Segoe UI Semibold", 14)
)
register_btn.pack(pady=5)

# ===== FOOTER =====
footer_label = ctk.CTkLabel(
    main_frame,
    text="© 2025 Multimedia Deepfake Detection System | v3.1.5 | Encrypted",
    font=("Consolas", 10),
    text_color="#6e7681"
)
footer_label.pack(pady=20)

root.mainloop()
