import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import UNET_OXFORD_IIIT
from loss import sparse_categorical_cross_entropy_object

# Normalize function
def normalize_oxford_iiit(input_image, mask_label, image_size):
    input_image = input_image.resize((image_size, image_size), Image.BILINEAR)
    mask_label = mask_label.resize((image_size, image_size), Image.NEAREST)
    input_image = np.array(input_image) / 255.0
    mask_label = np.array(mask_label).astype(np.int64) - 1  # classes: 1,2,3 → 0,1,2
    return input_image, mask_label

# Dataset
class OxfordPetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=128):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        
        # 이미지와 마스크의 공통된 파일 이름만 추림 (확장자 제외)
        image_files = [f for f in os.listdir(image_dir) if not f.startswith("._") and f.endswith(".jpg")]
        mask_files = [f for f in os.listdir(mask_dir) if not f.startswith("._") and f.endswith(".png")]

        image_ids = set([os.path.splitext(f)[0] for f in image_files])
        mask_ids = set([os.path.splitext(f)[0] for f in mask_files])
        common_ids = sorted(list(image_ids & mask_ids))

        self.images = [f"{id}.jpg" for id in common_ids]
        self.masks = [f"{id}.png" for id in common_ids]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image, mask = normalize_oxford_iiit(img, mask, self.image_size)
        image = Image.fromarray((image * 255).astype(np.uint8))  # for transform

        if self.transform:
            # seed = np.random.randint(2147483647)
            # torch.manual_seed(seed)
            image = self.transform(image)

        # 마스크는 변환 적용하지 않고 그대로 Tensor로 변환
        mask = torch.from_numpy(mask).long()

        return image, mask

# 시각화
def display(display_list, epoch=None):
    plt.figure(figsize=(15, 5))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        img = display_list[i]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        plt.imshow(img.squeeze(), None, vmin=0, vmax=2 if i != 0 else 1)
        plt.axis('off')
    if epoch is not None:
        plt.savefig(f'oxford_iiit_pet/epoch_{epoch}.jpg')
    plt.show()

def create_mask(pred_mask):
    pred_mask = torch.argmax(pred_mask, dim=1)
    return pred_mask[0].cpu().numpy()

def show_predictions(model, sample_image, sample_mask, device, epoch=None):
    model.eval()
    with torch.no_grad():
        pred = model(sample_image.unsqueeze(0).to(device))
        pred_mask = create_mask(pred)
    input_img = (sample_image.cpu().numpy() * 255).astype(np.uint8)
    true_mask = sample_mask.cpu().numpy()
    display([input_img, true_mask, pred_mask], epoch=epoch)

def main():
    # 하이퍼파라미터
    num_epochs = 20
    steps_per_epoch = 2000
    num_classes = 3
    batch_size = 16
    learning_rate = 0.001
    image_size = 128
    checkpoint_path = 'saved_model_oxford_iiit/unet_model.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)

    # Transform
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Dataset & Loader
    train_dataset = OxfordPetDataset(
        image_dir='./oxford_iiit_pet/images',
        mask_dir='./oxford_iiit_pet/annotations/trimaps',
        transform=transform,
        image_size=image_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True
    )

    # Model
    model = UNET_OXFORD_IIIT(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = sparse_categorical_cross_entropy_object

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_loss = float('inf')
    sample_image, sample_mask = next(iter(train_loader))
    sample_image, sample_mask = sample_image[0], sample_mask[0]
    show_predictions(model, sample_image, sample_mask, device, epoch=0)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for step, (images, masks) in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{step}/{steps_per_epoch}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Best model saved at epoch {epoch+1}")

    show_predictions(model, sample_image, sample_mask, device, epoch=epoch+1)

    print("🎉 Training complete.")

if __name__ == '__main__':
    main()
