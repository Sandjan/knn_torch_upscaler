import torch
import torch.nn.functional as F
import torchvision.transforms as T

@torch.jit.script
def signed_log(t,u:float=0.7):
    return torch.sign(t) * torch.pow(torch.abs(t),u)

def scale_up_pytorch(original:torch.Tensor, 
                     target_width:int, 
                     target_height:int, 
                     device:torch.device, 
                     a:float=7.55, 
                     b:float=4.69, 
                     c:float=0.1, 
                     d:float=1.46, 
                     e:float=0.8, 
                     r:int=4, 
                     t:int=1, 
                     u:float=7.898, 
                     k:int=7, 
                     s:float=1.1, 
                     v:float=10.0, 
                     w:int=30, 
                     z:float=0.268):
    w = int(w)
    if w % 2 == 0:
        w -= 1
    
    org_height, org_width = original.shape[1], original.shape[2]
    scale_y = target_height / org_height
    scale_x = target_width / org_width
    
    upscaled = F.interpolate(
        original.unsqueeze(0), 
        size=(target_height, target_width), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    gaussian_blur_layer_w = T.GaussianBlur(kernel_size=(w, w), sigma=z)
    blurred = gaussian_blur_layer_w(upscaled)
    
    orig = upscaled
    mask = orig - blurred
    upscaled_in = torch.clamp(orig + (mask) * v, 0, 255)

    original_height, original_width = original.shape[1], original.shape[2]
    up_height, up_width = upscaled_in.shape[1], upscaled_in.shape[2]
    
    gaussian_blur_layer = T.GaussianBlur(kernel_size=(k, k), sigma=s)
    downupscaled = gaussian_blur_layer(original)
    
    difference = (original - downupscaled)
    patch_size = 3
    dilation = t
    half_patch = (patch_size * dilation) // 2 - (t // 2)

    patches_up = F.unfold(
        upscaled_in.unsqueeze(0), 
        kernel_size=patch_size, 
        padding=half_patch, 
        dilation=dilation
    ).view(1, -1, up_height, up_width)
    patches_down = F.unfold(
        downupscaled.unsqueeze(0), 
        kernel_size=patch_size, 
        padding=half_patch, 
        dilation=dilation
    ).view(1, -1, original_height, original_width)

    result_diff = torch.zeros((3, up_height, up_width), dtype=torch.float16).to(device)
    weight_sum = torch.zeros((up_height, up_width), dtype=torch.float16).to(device)

    y_range = (torch.arange(up_height, dtype=torch.int16, device=device) // scale_y).int()
    x_range = (torch.arange(up_width, dtype=torch.int16, device=device) // scale_x).int()
    
    for y in range(-r, r):
        for x in range(-r, r):
            idx_y = torch.clip(y_range + y, 0, original_height - 1)
            idx_x = torch.clip(x_range + x, 0, original_width - 1)
            
            weigth = u / ((((patches_down[:, :, idx_y, :][:, :, :, idx_x] - patches_up).abs()).mean(dim=1).squeeze(0) / a) ** b + c)
            weight_sum += weigth
            result_diff += difference[:, idx_y, :][:, :, idx_x] * weigth

    upscaled_in += d * signed_log(result_diff / (7.7e-06 + weight_sum), e)

    return torch.clamp(upscaled_in, 0, 255)