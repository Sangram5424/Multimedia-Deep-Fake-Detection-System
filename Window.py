import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from tkvideo import tkvideo
import os
from subprocess import call

class DeepFakeLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimedia Deepfake Detection System")
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()

        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#0a0e17")

        self.setup_video_background()
        self.create_ui()

    def setup_video_background(self):
        try:
            self.video_label = tk.Label(self.root)
            self.video_label.place(x=0, y=0, relwidth=1, relheight=1)

            video_path = "C.mp4"  # Replace with your video path
            if os.path.exists(video_path):
                player = tkvideo(video_path, self.video_label, loop=1, size=(self.screen_width, self.screen_height))
                player.play()
            else:
                self.root.configure(bg="#0a0e17")
                tk.Label(self.root, text="Video not found", fg="white", bg="#0a0e17", font=("Helvetica", 18)).pack(pady=20)
        except Exception as e:
            print("Error loading video:", e)

    def create_ui(self):
        # Main transparent frame
        self.frame = tk.Frame(self.root, bg='#0d1117')
        self.frame.place(relx=0.5, rely=0.8, anchor='center')

        # Start Button
        self.start_button = tk.Button(
            self.frame,
            text="START",
            command=self.start_analysis,
            font=("Helvetica", 20, "bold"),
            bg="white",
            fg="black",
            activebackground="#d0d0d0",
            bd=0,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        self.start_button.pack()

    def start_analysis(self):
        try:
            self.root.destroy()
            call(["python", "GUI_main.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start analysis:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DeepFakeLauncher(root)
    root.mainloop()
