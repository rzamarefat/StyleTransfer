import torch 
import torchvision
import numpy as np
from Backbone import Backbone
from utils import *
from tqdm import tqdm 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--path-to-original-image',  type=str, help='Path to the original image')
    parser.add_argument('--path-to-style-image',  type=str, help='Path to the style image')
    parser.add_argument('--path-to-save-output-image',  type=str, help='Path to save the output image')
    parser.add_argument('--num-steps',  default=10000, type=int, help='Number of steps to optimize')
    parser.add_argument('--alpha',  default=1, type=float, help='Alpha parameters')
    parser.add_argument('--beta',  default=1e-2, type=float, help='Beta parameters')
    parser.add_argument('--device', default='cuda', type=str, help='Device: GPU/CPU')
    parser.add_argument('--backbone', default='vgg19', type=str, help='Backbone name')


    # path_to_original_image = "/mnt/829A20D99A20CB8B/projects/github_projects/Style_Transfer/images/robot.jpg"
    # path_to_style_image = "/mnt/829A20D99A20CB8B/projects/github_projects/Style_Transfer/images/style_image_2.jpg"

    args = parser.parse_args()

    LEARNING_RATE = 1e-3
    STEPS_NUM = 10000
    ALPHA = args.alpha
    BETA = args.beta

    device = args.device

    backbone = Backbone(device=device, backbone_name="VGG19")

    image_size = 356

    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((image_size, image_size))
    ])


    original_img = read_img(path=path_to_original_image, transforms=transforms, device=device)
    style_img = read_img(path=path_to_style_image, transforms=transforms, device=device)

    generated_img = original_img.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([generated_img], lr=LEARNING_RATE)

    for step in tqdm(range(args.num_steps)):
        original_feats = backbone(original_img)
        style_feats = backbone(style_img)
        generated_feats = backbone(generated_img)

        style_loss = 0
        content_loss = 0

        for ori_feat, st_feat, gen_feat in zip(original_feats, style_feats, generated_feats):
            batch_size, c, w, h = gen_feat.shape

            content_loss += torch.mean((gen_feat - ori_feat) ** 2)

            G = gen_feat.view(c, w*h).mm(
                gen_feat.view(c, w*h).t()
            )

            A = st_feat.view(c, w*h).mm(
                st_feat.view(c, w*h).t()
            )

            style_loss += torch.mean((G - A) ** 2)


        total_loss = ALPHA * content_loss + BETA * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Total Loss: {total_loss}")
            print(generated_img.shape)
            save_img(generated_img, args.path_to_save_output_image)