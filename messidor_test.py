from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from uploading import messidor_test_loader

device = torch.device("cpu")

# model (5 klas)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 5)

model.load_state_dict(torch.load("ddr_model_cpu3.pth", map_location=device))
model.to(device)
model.eval()

# collect predictions
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in messidor_test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        # prawdopodobieństwo choroby - suma prawdopodobieństw klas 1-4
        disease_prob = probs[:, 1:].sum(dim=1)
        threshold = 0.3
        preds = (disease_prob > threshold).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
# 0 - zdrowy, 1–4 - chory
binary_preds = [0 if p == 0 else 1 for p in all_preds]
binary_labels = [0 if y == 0 else 1 for y in all_labels]

# confusion matrix
cm = confusion_matrix(binary_labels, binary_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Healthy", "Disease"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Messidor Confusion Matrix (Binary: 0 vs 1-4)")
plt.show()