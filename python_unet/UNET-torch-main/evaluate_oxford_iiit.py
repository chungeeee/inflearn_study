import os
import argparse
import torch
import numpy as np
from skimage.io import imsave
from model import UNET_OXFORD_IIIT
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path

# 설정 값
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='saved_model_oxford_iiit/unet_model.pth',
                        help='path to a directory to restore checkpoint file')
    parser.add_argument('--test_dir', type=str, default='oxford_iiit_test_result',
                        help='directory where test prediction result will be saved')
    parser.add_argument('--num_classes', type=int, default=3, help='number of prediction classes')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--total_test_image_num', type=int, default=30)
    parser.add_argument('--test_img_dir', type=str, default='./oxford_iiit_pet/images')
    parser.add_argument('--image_size', type=int, default=128, help='Size of the input image')
    return parser.parse_args()

# Normalize function
def normalize_oxford_iiit(input_image, image_size):
    input_image = input_image.resize((image_size, image_size), Image.BILINEAR)
    input_image = np.array(input_image) / 255.0
    return input_image

def create_mask(pred_mask):
    pred_mask = torch.argmax(pred_mask, dim=1)  # 확률 벡터에서 가장 큰 값의 인덱스를 찾기
    return pred_mask.cpu().numpy()

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None, image_size=128):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_size = image_size

        # 이미지 파일 리스트 가져오기
        image_files = [f for f in self.image_dir.glob('*.jpg') if not f.name.startswith("._")]
        self.images = sorted([f.name for f in image_files])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.image_dir / self.images[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None
        
        image = normalize_oxford_iiit(img, self.image_size)
        image = Image.fromarray((image * 255).astype(np.uint8))  # for transform

        if self.transform:
            image = self.transform(image)
        else:
            # 기본적으로 ToTensor()를 추가
            image = transforms.ToTensor()(image)

        return image

def main():
    args = parse_args()

    # 체크포인트 파일이 존재하는지 확인
    if not os.path.exists(args.checkpoint_path):
        print('Checkpoint file does not exist!')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNET_OXFORD_IIIT(args.num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()
    print(f'{args.checkpoint_path} checkpoint is restored!')

    # 테스트 데이터셋 로딩
    test_dataset = TestDataset(args.test_img_dir, transform=transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()  # 이미지를 Tensor로 변환
    ]))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 결과 저장 디렉토리 생성
    os.makedirs(args.test_dir, exist_ok=True)
    print(f'Total test images: {min(args.total_test_image_num, len(test_loader))}')

    with torch.no_grad():
        for image_num, test_image in enumerate(test_loader):
            if image_num >= args.total_test_image_num:
                break

            if test_image is None:  # 오류 처리: None 값 처리
                continue

            test_image = test_image.to(device)
            pred_mask = model(test_image)
            mask_img = create_mask(pred_mask)
            
            output_image_path = os.path.join(args.test_dir, f'{image_num}_result.png')
            imsave(output_image_path, (mask_img * 255).astype(np.uint8))
            print(f'{output_image_path} saved!')

if __name__ == '__main__':
    main()
