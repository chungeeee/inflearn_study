import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from model import UNET_ISBI_2012
from loss import binary_loss_object

# 데이터 정규화 함수
def normalize_isbi_2012(input_image, mask_label):
    input_image = np.array(input_image) / 255.0
    mask_label = np.array(mask_label) / 255.0
    mask_label = (mask_label > 0.5).astype(np.float32)
    return input_image, mask_label

# 데이터셋 클래스
class ISBIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        image, mask = normalize_isbi_2012(image, mask)
        image = Image.fromarray((image * 255).astype(np.uint8))
        mask = Image.fromarray((mask * 255).astype(np.uint8))

        if self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        return image, mask.squeeze(0)  # 채널 차원 제거

# 시각화 함수
def display(display_list, epoch=None):
    plt.figure(figsize=(15, 5))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        img = display_list[i]
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = img.squeeze()
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')
    if epoch is not None:
        plt.savefig(f'saved_model_isbi_2012/epoch_{epoch}.jpg')
    plt.show()

def create_mask(pred_mask):
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    return pred_mask[0].cpu().numpy()

def show_predictions(model, sample_image, sample_mask, device, epoch=None):
    model.eval()
    with torch.no_grad():
        pred = model(sample_image.unsqueeze(0).to(device))
        pred_mask = create_mask(pred)
    input_img = (sample_image.cpu().numpy() * 255).astype(np.uint8)
    true_mask = (sample_mask.cpu().numpy() * 255).astype(np.uint8)
    display([input_img, true_mask, pred_mask * 255], epoch=epoch)

def main():
    # 하이퍼파라미터 설정
    batch_size = 2
    learning_rate = 1e-4
    num_epochs = 5
    steps_per_epoch = 2000
    num_classes = 1
    checkpoint_path = 'saved_model_isbi_2012/unet_model.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)

    # 데이터 증강
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(512, scale=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # 데이터로더
    train_dataset = ISBIDataset(
        image_dir='./isbi_2012/preprocessed/train_imgs',
        mask_dir='./isbi_2012/preprocessed/train_labels',
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        drop_last=True,
        pin_memory=True
    )

    # 모델 초기화
    model = UNET_ISBI_2012(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = binary_loss_object

    # 체크포인트 디렉토리 생성
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # 학습 루프
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
            masks = masks.to(device).unsqueeze(1)

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

        # 체크포인트 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved at epoch {epoch+1}")

    # 예측 결과 시각화
    show_predictions(model, sample_image, sample_mask, device, epoch=epoch+1)

    print("Training complete.")

if __name__ == '__main__':
    main()