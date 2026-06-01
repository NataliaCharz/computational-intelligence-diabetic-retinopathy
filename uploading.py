import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # losowe odbicie poziome
    transforms.RandomHorizontalFlip(),
    # losowe obroty do 10 stopni
    transforms.RandomRotation(10),
    # losowe zmiany jasności, kontrastu i nasycenia
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    # tensor i normalizacja - te same wartości co ImageNet (ResNet był trenowany na ImageNet)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(
    "data/DDR/DDR/512/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    "data/DDR/DDR/512/valid",
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    "data/DDR/DDR/512/test",
    transform=val_test_transform
)

messidor_test = datasets.ImageFolder(
    "data/messidor/test",
    transform=val_test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16
)

messidor_test_loader = DataLoader(
    messidor_test,
    batch_size=32,
    shuffle=False
)
