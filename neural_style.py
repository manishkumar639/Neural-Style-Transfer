import torch
from torchvision import models, transforms
from core.helpers import load_and_preprocess, tensor_to_img
import os
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model = models.vgg19(pretrained=True).features.to(DEVICE).eval()

for param in vgg_model.parameters():
    param.requires_grad_(False)

def extract_layers(tensor, model):
    selected = {
        '0': 'l1', '5': 'l2',
        '10': 'l3', '19': 'l4',
        '21': 'main', '28': 'l5'
    }
    features = {}
    for name, layer in model._modules.items():
        tensor = layer(tensor)
        if name in selected:
            features[selected[name]] = tensor
    return features

def compute_gram(tensor):
    _, channels, height, width = tensor.shape
    reshaped = tensor.view(channels, height * width)
    return torch.mm(reshaped, reshaped.t())

def generate_styled_image(base_path, art_path, output_dir='artifacts', steps=400, update_step=400):
    os.makedirs(output_dir, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize(300),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    base_tensor = load_and_preprocess(base_path, tfm, DEVICE)
    style_tensor = load_and_preprocess(art_path, tfm, DEVICE)
    result = base_tensor.clone().requires_grad_(True)

    style_layers = {
        'l1': 1.0, 'l2': 0.75, 'l3': 0.5,
        'l4': 0.25, 'l5': 0.1
    }

    weight_content = 150
    weight_style = 1e8

    content_feats = extract_layers(base_tensor, vgg_model)
    style_feats = extract_layers(style_tensor, vgg_model)
    gram_style = {k: compute_gram(v) for k, v in style_feats.items()}

    optimizer = torch.optim.Adam([result], lr=0.007)

    for step in range(1, steps + 1):
        output_feats = extract_layers(result, vgg_model)

        c_loss = torch.mean((output_feats['main'] - content_feats['main'])**2)

        s_loss = 0
        for key in style_layers:
            G = compute_gram(output_feats[key])
            A = gram_style[key]
            _, c, h, w = output_feats[key].shape
            s_loss += style_layers[key] * torch.mean((G - A)**2) / (c * h * w)

        total = weight_content * c_loss + weight_style * s_loss

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        if step % update_step == 0 or step == steps:
            output_arr = tensor_to_img(result)
            file_path = os.path.join(output_dir, f"fusion_result_{step}.png")
            plt.imsave(file_path, output_arr)
            print(f"[{step}/{steps}] -> Loss: {total:.4f}")
    return file_path
