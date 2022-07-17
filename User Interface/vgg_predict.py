import torch
from PIL import Image
from DataPreprocess import data_transform
from VGG19 import VGG19
import numpy as np


def softmax(inMatrix):
    m, n = np.shape(inMatrix)
    outMatrix = np.mat(np.zeros((m, n)))
    soft_sum = 0
    for idx in range(0, n):
        outMatrix[0, idx] = np.exp(inMatrix[0, idx])
        soft_sum += outMatrix[0, idx]
    for idx in range(0, n):
        outMatrix[0, idx] = round(outMatrix[0, idx] / soft_sum, 3)
    return outMatrix


def makedecision(output):
    if output == 0:
        types = "COVID"
    elif output == 1:
        types = "Lung Opacity"
    elif output == 2:
        types = "Normal"
    elif output == 3:
        types = "Viral Pneumonia"
    return types


def predict(img):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGG19()
    print("******************************************************")
    print("VGG Model Initialize Successfully!")

    model.to(device)
    model.load_state_dict(torch.load("./vgg19_final_project.pth"))
    model.eval()
    print("******************************************************")
    print("Model Parameters Loaded")

    img = Image.open(img)
    img = data_transform(img)
    img = img.view(1, 1, 224, 224)
    img = img.to(device)
    output = model(img)
    print(output)

    prob = softmax(output.cpu().detach().numpy())
    _, prediction = torch.max(output, 1)
    prediction = prediction[0].item()
    prob = str(float(prob[:, prediction]) * 100) + "%"
    types = makedecision(prediction)
    print("******************************************************")
    print(f"The probability of you are diagnosed with {types} is {prob}")

    return prob, types


if __name__ == '__main__':
    img = Image.open("./test/0/COVID-702.png")
    predict(img)
