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

img = None
# Function to generate images using the selected image path.
def generate_images_wrapper():
    global img
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;")])
    if file_path:
        img = Image.open(file_path)

    # If the image is smaller than 300x300, resize it to fit 300x300
    if img.width < 300 or img.height < 300:
        img = img.resize((300, 300))
        # Display the selected image in the Tkinter app
    img.thumbnail((300, 300))
    photo = ImageTk.PhotoImage(img)

    selected_image_label.config(image=photo)
    selected_image_label.image = photo

    # Clear generated photo
    generated_image_label.config(image=None)
    generated_image_label.image = None

    # Store the selected image path in a variable to be used later
    generate_images_wrapper.selected_image_path = file_path

# Function to generate the image and display scores when the "Generate" button is clicked
def generate_button_click():
    global img
    try:
        if (isinstance(seed_label_input.get(), int) == False):
            text.delete("1.0", "end")
            text.insert("insert", "Seed must be an integer!")
    except:
        pass

    if hasattr(generate_images_wrapper, "selected_image_path"):
        file_path = generate_images_wrapper.selected_image_path

        if file_path:
            if v.get() == "reconstruct":
                if seed_label_input.get() == '':
                    seed = None
                else:
                    seed = int(seed_label_input.get())
                gen_image = generate_image.generate_images(file_path, seed)
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

            # Clear the existing text before inserting new scores
            text.delete("1.0", "end")

            text.insert("insert", "PSNR: " + str(psnr_score))
            text.insert("insert", "\nSSIM: " + str(ssim_score))
    else:
        # No image selected, display an error message
        text.delete("1.0", "end")
        text.insert("insert", "Please select an image first!")

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

# Create a frame to hold the image labels
image_frame = tk.Frame(root)
image_frame.pack(pady=10)

# Create a label to show the selected image
selected_image_label = tk.Label(image_frame)
selected_image_label.grid(row=0, column=0)  # Place the selected image label in the first column

# Create a label to display the generated image
generated_image_label = tk.Label(image_frame)
generated_image_label.grid(row=0, column=1)  # Place the generated image label in the second column

# Seed Input
seed_label = tk.Label(root, text="Enter Seed Number").pack()
seed_label_input=tk.Entry(root, width=35)
seed_label_input.pack()
# Create the "Generate" button
generate_button = tk.Button(root, text="Generate", command=generate_button_click)
generate_button.pack(pady=5)

# Display PSNR score
text = tk.Text(root)
text.pack(side="bottom")


root.mainloop()
