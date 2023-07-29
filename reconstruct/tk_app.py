import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import generate_image

# Import the 'generate_images' function from your script and complete it with all dependencies.

# Function to generate images using the selected image path.
def generate_images_wrapper():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;")])
    if file_path:
        image = Image.open(file_path)
        # Display the selected image in the Tkinter app
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)

        selected_image_label.config(image=photo)
        selected_image_label.image = photo

        gen_image = generate_image.generate_images(file_path)

        # Display the generated image in the Tkinter app.
        gen_image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(gen_image)

        generated_image_label.config(image=photo)
        generated_image_label.image = photo


# Create the main application window
root = tk.Tk()
root.title("Image Generation App")
root.geometry("1000x800")

# Create a button to open the file dialog
open_button = tk.Button(root, text="Select Image", command=generate_images_wrapper)
open_button.pack(pady=5)

# Create a label to show the selected image
selected_image_label = tk.Label(root)
selected_image_label.pack(pady=10)



# Create a label to display the generated image
generated_image_label = tk.Label(root)
generated_image_label.pack(pady=10)

root.mainloop()
