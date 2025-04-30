import os
import pickle
import math
import torch
import torch.nn.functional as F
import torchvision
# from datasets import get_dataset
# from safetensors import safe_open
from skimage.measure import regionprops
from skimage.morphology import label
from torchvision.transforms.functional import gaussian_blur
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt

def make_square_bbox(minr, minc, maxr, maxc, max_height=224, max_width=224, min_area=500):
    height = maxr - minr
    width = maxc - minc
    size = max(height, width)

    # 최소 면적 확보
    area = height * width
    if size * size < min_area:
        size = math.ceil(math.sqrt(min_area))

    center_r = (minr + maxr) // 2
    center_c = (minc + maxc) // 2

    half_size = size // 2

    minr = max(0, center_r - half_size)
    maxr = min(max_height, center_r + size - half_size)
    minc = max(0, center_c - half_size)
    maxc = min(max_width, center_c + size - half_size)

    # 경계에 걸린 경우 재조정
    if minr == 0:
        maxr = min(max_height, minr + size)
    elif maxr == max_height:
        minr = max(0, maxr - size)

    if minc == 0:
        maxc = min(max_width, minc + size)
    elif maxc == max_width:
        minc = max(0, maxc - size)

    return int(minr), int(minc), int(maxr), int(maxc)

def apply_mask_with_border(img, mask, alpha, mask_th, draw_border=True):
    # 마스크를 이진화
    binary_mask = (mask >= mask_th).float()

    # 경계 검출을 위한 컨볼루션 커널
    kernel = torch.tensor([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 패딩 추가 및 경계 검출
    padded_mask = F.pad(binary_mask.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
    edges = F.conv2d(padded_mask, kernel).squeeze()
    edges = (edges != 0).float()

    # 마스크의 역을 계산
    inv_mask = 1 - binary_mask

    # 알파 블렌딩 적용
    blended_img = img * binary_mask.unsqueeze(0) + img * inv_mask.unsqueeze(0) * alpha

    # 경계를 검은색으로 설정
    if draw_border:
        black_color = torch.tensor([0.0, 0.0, 0.0]).view(3, 1, 1)
        blended_img = torch.where(edges.unsqueeze(0) == 1, black_color, blended_img)

    return blended_img

def crop_mask(img, mask, crop_th=0.8, mask_th=None, alpha=0.5, draw_border=True):
    if mask_th is None:
        mask_th = crop_th - 0.4

    # Resize
    resize_func = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
    ])

    if mask.size(-1) != img.size(-1):
        mask = resize_func(mask.unsqueeze(0))[0]

    # Blur
    mask_blur = gaussian_blur(mask.unsqueeze(0), kernel_size=51)[0]
    mask_blur = mask_blur.abs() / (mask_blur.abs().max() + 1e-8)

    # Crop
    mask_binary = mask_blur.clone()
    mask_binary[mask_binary < crop_th] = 0
    mask_binary[mask_binary >= crop_th] = 1

    # Bounding boxes
    label_img = label(mask_binary)
    regions = regionprops(label_img)

    # Get largest box
    regions = sorted(regions, key=lambda x: x.area, reverse=True)
    if len(regions) == 0:
        minr, minc, maxr, maxc = 0, 0, img.shape[-2], img.shape[-1]
    else:
        minr, minc, maxr, maxc = regions[0].bbox

    # 정사각형으로 만들기
    minr, minc, maxr, maxc = make_square_bbox(minr, minc, maxr, maxc)
    cropped = resize_func(img[:, minr:maxr, minc:maxc])
    mask_cropped = resize_func(mask_blur[minr:maxr, minc:maxc].unsqueeze(0))[0]

    cropped = apply_mask_with_border(cropped, mask_cropped, alpha=alpha, mask_th=mask_th, draw_border=draw_border)

    return cropped

def draw_channel_images(img_list, mask_list, option, save_name, crop_th=0.8, mask_th=0.4, alpha=0.5):
    # option = 'original_image'  # 'cropped_image', 'mask'
    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows),
                                gridspec_kw={'wspace': 0, 'hspace': 0})
    axes = axes.ravel()
    for i in range(len(axes)):
        img = img_list[i]
        mask = mask_list[i]
        if option == 'original_image':
            axes[i].imshow(img.permute(1, 2, 0))
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        elif option == 'mask':
            axes[i].imshow(mask)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        elif option == 'cropped_image':
            cropped = crop_mask(img, mask, crop_th=crop_th, mask_th=mask_th, alpha=alpha)
            axes[i].imshow(cropped.permute(1, 2, 0))
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    plt.savefig(save_name)
    plt.show()