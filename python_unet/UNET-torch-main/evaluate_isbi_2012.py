import os
import argparse
import torch
import numpy as np
from skimage.io import imsave
from model import UNET_ISBI_2012
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='saved_model_isbi_2012/unet_model.pth',
                        help='path to a directory to restore checkpoint file')
    parser.add_argument('--test_dir', type=str, default='isbi_2012_test_result',
                        help='directory which test prediction result saved')
    parser.add_argument('--num_classes', type=int, default=1, help='number of prediction classes')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--total_test_image_num', type=int, default=30)
    parser.add_argument('--test_img_dir', type=str, default='./isbi_2012/preprocessed/test_imgs')
    return parser.parse_args()

def normalize_isbi_2012(input_images):
    return input_images / 255.0

def create_mask(pred_mask):
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    return pred_mask[0].cpu().numpy()  # [1,512,512] → [512,512]

class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = sorted(os.listdir(img_dir))
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.img_list[idx])).convert('L')
        img = np.array(img, dtype=np.float32)
        img = normalize_isbi_2012(img)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1,H,W]
        return img

def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint_path):
        print('checkpoint file does not exist!')
        exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNET_ISBI_2012(args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    print(f'{args.checkpoint_path} checkpoint is restored!')

    test_dataset = TestDataset(args.test_img_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs(args.test_dir, exist_ok=True)
    print('total test image :', args.total_test_image_num)

    with torch.no_grad():
        for image_num, test_image in enumerate(test_loader):
            if image_num >= args.total_test_image_num:
                break
            test_image = test_image.to(device)
            pred_mask = model(test_image)
            mask_img = create_mask(pred_mask)
            output_image_path = os.path.join(args.test_dir, f'{image_num}_result.png')
            imsave(output_image_path, (mask_img * 255).astype(np.uint8))
            print(f'{output_image_path} saved!')

if __name__ == '__main__':
    main()