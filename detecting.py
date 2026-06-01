import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from uploading import messidor_test_loader

device = torch.device("cpu")
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load("ddr_model_cpu3.pth", map_location=device))
model = model.to(device)
model.eval()
# Grad-CAM dla ostatniej warstwy konwolucyjnej
target_layers = [model.layer4[-1]]
# Inicjalizacja Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)

class_names = ["0", "1", "2", "3", "4"]

for batch_idx, (images, labels) in enumerate(messidor_test_loader):
    images = images.to(device)
    # Analiza pierwszych 3 obrazów z batcha
    for i in range(min(3, images.size(0))):
        # Przygotowanie obrazu do Grad-CAM
        input_tensor = images[i].unsqueeze(0)
        # Predykcja modelu i obliczenie prawdopodobieństw
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            # Klasa z najwyższym prawdopodobieństwem
            pred_class = torch.argmax(probs, dim=1).item()
            disease_prob = probs[:, 1:].sum(dim=1).item()
            pred_binary = int(disease_prob > 0.3)
        true_class = labels[i].item()
        true_binary = 0 if true_class == 0 else 1
        # Generowanie mapy Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        # Normalizacja obrazu do zakresu [0, 1] dla wizualizacji
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        # Nałożenie mapy Grad-CAM na obraz
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        plt.figure(figsize=(6, 6))
        plt.imshow(visualization)
        plt.axis("off")
        plt.title(
            f"Pred: {pred_class} ({class_names[pred_class]}) | "
            f"Binary: {pred_binary} | "
            f"True: {true_binary}"
        )
        plt.show()
    if batch_idx == 10:
        break