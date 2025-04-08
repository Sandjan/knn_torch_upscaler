from knn_torch_upscaler import scale_up_pytorch
import cv2
import torch
import numpy as np

image_path = input("Input image path:")
iterations = [2,4,6][int(input("""What is your target scale?
1. x2
2. x4
3. x8
:"""))-1]
radius = 4
scale = 1.414213562

original = cv2.imread(image_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
original = torch.tensor(original, dtype=torch.float16).permute(2,0,1).to(device)

for i in range(iterations):
    org_height, org_width = original.shape[1], original.shape[2]
    upscaled = scale_up_pytorch(original,int(org_width*scale+0.5),int(org_height*scale+0.5),device)
    original = upscaled

upscaled = original.permute(1,2,0).cpu().numpy().astype(np.uint8)
cv2.imwrite('upscaled.png', upscaled)