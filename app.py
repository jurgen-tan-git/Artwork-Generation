import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

# Define the transformations to be applied to the input image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the pre-trained CycleGAN model for style transfer
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.jit.load('checkpoint/CycleGAN_Generator_50.pt')
    return model

def predict(im, Gen_BA): 
    w, h = im.size
    if h or w >= 500:
        scale_factor = 500 / max(h, w)
        height = int(h * scale_factor)
        width = int(w * scale_factor)
        im = transforms.Resize((height, width))(im)

    input = transform(im)
    Gen_BA.eval()
    output = Gen_BA(input.unsqueeze(0))
    output = output / 2 + 0.5
    return (transforms.ToPILImage()(output.squeeze(0)))

# Set the page layout to wide mode
st.set_page_config(layout="wide")

# Header
st.title("[Vincent van Gogh](https://en.wikipedia.org/wiki/Vincent_van_Gogh) Style Transfer using CycleGAN")
st.markdown("""
    This app has been built using a machine learning model called CycleGAN. For more information on CycleGAN, 
    you can read this [paper](https://arxiv.org/abs/1703.10593). The dataset used has been sourced from 
    [UC Berkeley](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) and other open sourced datasets on Kaggle.
""")

# Information about Van Gogh in the sidebar
with st.sidebar:
    st.title("Vincent van Gogh - The Artistic Genius")
    st.image("vangogh.jpg", caption="Self-portrait by Vincent van Gogh, 1887", use_column_width=True)
    st.markdown("""
        Vincent van Gogh (1853 - 1890) was a Dutch post-impressionist painter and one of the most renowned artists in the history of Western art. 
        His unique style, characterized by bold colors and expressive brushwork, has left an indelible mark on the art world.
        
        * Early Life: Vincent van Gogh was born in the Netherlands and initially pursued careers as an art dealer and a missionary before dedicating himself to art.
        * Artistic Struggles: Van Gogh battled mental health issues throughout his life, and his struggles were often reflected in his paintings.
        * Artistic Legacy: Although van Gogh's work was not widely recognized during his lifetime, he has since become one of the most celebrated artists of all time.
        * Famous Works: Some of his most famous works include "Starry Night," "Sunflowers," "Irises," and numerous self-portraits.
        
        Vincent van Gogh's masterpieces continue to captivate audiences worldwide, and his legacy lives on as an inspiration to countless artists and art enthusiasts.
        
        Click [here](https://vangoghexpo.com/) to find out more about Van Gogh: The Immersive Experience.
    """)
    
# Upload an image for style transfer
file = st.file_uploader(label="Upload your image", type=['.png', '.jpg', 'jpeg'])

if file:
    image = file.read()
    st.caption("Your image.")

    # Display uploaded image and generated image side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        pred_button = st.button("Generate")
    with col2:
        if pred_button:
            # Load the pre-trained model
            Gen_BA = load_model()

            with st.spinner("Generating. Please wait..."):
                img = Image.open(io.BytesIO(image)).convert("RGB")
                gen_image = predict(img, Gen_BA)
                st.image(gen_image, caption="Generated Image", use_column_width=True)

            # Clean up memory
            del Gen_BA, img, gen_image, image
            torch.cuda.empty_cache()
