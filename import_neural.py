import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
from core.processor import generate_styled_image

class StyleFusionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Style Fusion - Artistic Neural Transfer")
        self.root.geometry("1024x720")

        self.source_img_path = None
        self.art_img_path = None
        self.result_img_path = None

        self.build_interface()

    def build_interface(self):
        tk.Label(self.root, text="Style Fusion", font=("Georgia", 24)).pack(pady=12)

        btns = tk.Frame(self.root)
        btns.pack(pady=10)

        tk.Button(btns, text="Select Source Impi page", command=self.pick_source).grid(row=0, column=0, padx=10)
        tk.Button(btns, text="Select Style Reference", command=self.pick_art).grid(row=0, column=1, padx=10)
        tk.Button(btns, text="Transfer Style", command=self.start_transfer).grid(row=0, column=2, padx=10)

        self.canvas = tk.Frame(self.root)
        self.canvas.pack(pady=20)

        self.lbl1 = tk.Label(self.canvas, text="Original")
        self.lbl2 = tk.Label(self.canvas, text="Style")
        self.lbl3 = tk.Label(self.canvas, text="Generated")

        self.lbl1.grid(row=0, column=0)
        self.lbl2.grid(row=0, column=1)
        self.lbl3.grid(row=0, column=2)

        self.panel1 = tk.Label(self.canvas)
        self.panel2 = tk.Label(self.canvas)
        self.panel3 = tk.Label(self.canvas)

        self.panel1.grid(row=1, column=0, padx=15)
        self.panel2.grid(row=1, column=1, padx=15)
        self.panel3.grid(row=1, column=2, padx=15)

    def pick_source(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if path:
            self.source_img_path = path
            self.display(path, self.panel1)
            messagebox.showinfo("Loaded", f"Source image selected:\n{os.path.basename(path)}")

    def pick_art(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if path:
            self.art_img_path = path
            self.display(path, self.panel2)
            messagebox.showinfo("Loaded", f"Style reference selected:\n{os.path.basename(path)}")

    def start_transfer(self):
        if not self.source_img_path or not self.art_img_path:
            messagebox.showerror("Input Error", "Please load both source and style images.")
            return

        def task():
            self.root.title("Processing Style Transfer...")
            self.toggle_all_buttons("disabled")
            try:
                out_file = generate_styled_image(self.source_img_path, self.art_img_path)
                self.result_img_path = out_file
                self.display(out_file, self.panel3)
                messagebox.showinfo("Success", f"Stylized image saved:\n{out_file}")
            except Exception as error:
                messagebox.showerror("Error", str(error))
            finally:
                self.toggle_all_buttons("normal")
                self.root.title("Style Fusion - Artistic Neural Transfer")

        threading.Thread(target=task).start()

    def toggle_all_buttons(self, state):
        for child in self.root.winfo_children():
            for btn in child.winfo_children():
                if isinstance(btn, tk.Button):
                    btn.config(state=state)

    def display(self, img_path, target_widget, size=(300, 300)):
        try:
            img = Image.open(img_path).resize(size)
            photo = ImageTk.PhotoImage(img)
            target_widget.config(image=photo)
            target_widget.image = photo
        except Exception as e:
            print("Display Error:", e)

if __name__ == "__main__":
    print("Starting GUI...")
    window = tk.Tk()
    app = StyleFusionApp(window)
    window.mainloop()
