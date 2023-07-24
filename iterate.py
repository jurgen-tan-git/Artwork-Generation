import torchvision.transforms as transforms
import torch
import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--inputfolder', type=str, required=True, help='input image folder to use')
args = parser.parse_args()

t = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
torch.set_default_device('mps')

# @st.cache_resource
def load_model():
    model = torch.jit.load('checkpoint/CycleGAN_Generator_50.pt')
    # model = torch.load('checkpoint/latest_net_G.pt')
    return model

Gen_BA = load_model()


def predict(im,Gen_BA): 
    w,h = im.size
    if h or w >=500:
        scale_factor = 500/max(h,w)
        height = int(h * scale_factor)
        width = int(w * scale_factor)
        im = transforms.Resize((height,width))(im)

    input = t(im)
    Gen_BA.eval()
    output = Gen_BA(input.unsqueeze(0))
    output = output/2 +0.5
    return (transforms.ToPILImage()(output.squeeze(0)))

folderpath = args.inputfolder
print(folderpath)
imagepaths = os.listdir(folderpath)

try:
    imagepaths.remove('.DS_Store')
except:
    pass

count = 0
for image in imagepaths:
    img = Image.open(folderpath + '/'+ image).convert("RGB")
    gen_image = predict(img,Gen_BA)
    gen_image.save( './outputs/'+ 'gen_' + image)
    count += 1
    print(count,len(os.listdir(folderpath)))
    
