import cv2
import numpy as np
import torch
from torchvision import transforms
from model import MyMLP

img = cv2.imread('4.jpg', cv2.IMREAD_GRAYSCALE)
img = np.invert(img)
img = cv2.resize(img, (28, 28))

img = transforms.ToTensor()(img)

model = MyMLP()
path = './my-mlp.pth'
model.load_state_dict(torch.load(path))
model.eval()


arr = img.numpy()
for y in range(arr.shape[1]):
    for x in range(arr.shape[2]):
        if arr[0, y, x] > 0:
            print(f'\033[32m{arr[0, y, x]:.2f}\033[0m ', end='')
        else:
            print(f'{arr[0, y, x]:.2f} ', end='')
    print()


with torch.no_grad() :
    outputs = model(img)
    _, pred = torch.max(outputs, dim=1)
    print(f"out model's prediction is {pred[0]}")