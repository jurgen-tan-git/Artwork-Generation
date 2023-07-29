import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import generate_image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Import the 'generate_images' function from your script and complete it with all dependencies.

# Function to get the Peak Signal Noise Ratio
def get_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2)

def get_ssim(img1, img2):
    return structural_similarity(img1, img2, data_range=255, channel_axis=2, multichannel=True)

# Function to generate images using the selected image path.
def generate_images_wrapper():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;")])
    if file_path:
        img = Image.open(file_path)
        # Display the selected image in the Tkinter app
        img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(img)

        selected_image_label.config(image=photo)
        selected_image_label.image = photo

        if v.get() == "reconstruct":
            gen_image = generate_image.generate_images(file_path)
        else:
            gen_image = generate_image.generate_style_image(file_path)

        # Display the generated image in the Tkinter app.
        gen_image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(gen_image)

        generated_image_label.config(image=photo)
        generated_image_label.image = photo

        img_resized = np.asarray(img.resize((512, 512)))
        gen_image_resized = np.asarray(gen_image.resize((512, 512)))
        psnr_score = get_psnr(img_resized, gen_image_resized)
        ssim_score = get_ssim(img_resized, gen_image_resized)
        text.insert("insert", "PSNR: " + str(psnr_score))
        text.insert("insert", "\nSSIM: " + str(ssim_score))


# Create the main application window
root = tk.Tk()
root.title("Image Generation App")
root.geometry("1000x800")

# Variable to store the selected model option
v = tk.StringVar(root, "style")

radioFrame = tk.Frame(root)
radioFrame.pack(pady=5)

styleBtn = tk.Radiobutton(radioFrame, text="Style Transfer", variable=v, value="style")
styleBtn.pack(side="left")

reconstructBtn = tk.Radiobutton(radioFrame, text="Reconstruction", variable=v, value="reconstruct")
reconstructBtn.pack(side="left")

# Create a button to open the file dialog
open_button = tk.Button(root, text="Select Image", command=generate_images_wrapper)
open_button.pack(pady=5)

# Create a label to show the selected image
selected_image_label = tk.Label(root)
selected_image_label.pack(pady=10)



# Create a label to display the generated image
generated_image_label = tk.Label(root)
generated_image_label.pack(pady=10)

# Display PSNR score

text = tk.Text(root)
text.pack(side="bottom")


root.mainloop()
