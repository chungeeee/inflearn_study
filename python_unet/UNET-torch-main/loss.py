import torch.nn as nn

# Binary Cross Entropy loss for ISBI-2012 dataset (for binary classification tasks)
binary_loss_object = nn.BCEWithLogitsLoss()  # from_logits=False를 처리하기 위해 이 loss를 사용합니다.

# Sparse Categorical Cross Entropy loss for Oxford-IIIT dataset (for multi-class segmentation)
sparse_categorical_cross_entropy_object = nn.CrossEntropyLoss()  # from_logits=True가 기본값이라 별도 설정 필요 없음
