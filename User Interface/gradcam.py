import os
import numpy as np
import torch
from PIL import Image
from DataPreprocess import data_transform
from gradcam_utils import GradCAM, show_cam_on_image
from VGG19 import VGG19
import cv2


def heatmap(weight, img_path):
    # load model & target_layers
    model = VGG19()
    model.load_state_dict(torch.load(weight))
    target_layers = model.conv_list[-1]

    # image preprocess
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # GradCAM apply
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 0

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]

    # visualize
    visualization = show_cam_on_image(np.array(img).astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)

    # save images
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'gc_{os.path.basename(img_path)}', visualization)
    print(f'Heatmap Save to gc_{os.path.basename(img_path)}')


if __name__ == '__main__':
    heatmap("./vgg19_final_project.pth", "./test/0/COVID-851.png")
