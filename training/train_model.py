import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone
import logging
from contextlib import redirect_stdout
import os

from result import Pikmin2cResult
from models import ScoreRecognizer

DATA_DIR = Path(__file__).parent / 'data'
TRAIN_IMAGE_DIR = DATA_DIR / '2c'
OUTPUT_DIR = Path(__file__).parent / 'output'

BATCH_SIZE = 64
EPOCHS = 50

JST = timezone(timedelta(hours = +9), 'JST')
DATE_FORMAT = datetime.now(JST).strftime('%Y%m%d%H%M')
OUTPUT_PATH = OUTPUT_DIR / f'2c_{DATE_FORMAT}'

# ロガー
def create_logger(log_file_path: Path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # ファイル
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    # 標準出力
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # ログのフォーマット
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    def jst(*args):
        return datetime.now(JST).timetuple()
    formatter.converter = jst
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # ハンドラー追加
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
logger = create_logger(f'{OUTPUT_PATH}.log')

# データの準備
results: list[Pikmin2cResult] = []
for image_path in TRAIN_IMAGE_DIR.iterdir():
    if image_path.suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
        result = Pikmin2cResult(image_path)
        results.append(result)
logger.info('Creating Training Data...')
images, labels = Pikmin2cResult.create_data_pairs(results)
logger.info('Done!')
dataset = torch.utils.data.TensorDataset(images, labels)
train_size = int(0.8 * len(dataset))
train_dataset, valid_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE)

model = ScoreRecognizer(11)
optimizer = Adam(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()

logger.info(f'num_params: {sum(p.numel() for p in model.parameters())}')

best_valid_loss = 1e6

for epoch in range(1, EPOCHS + 1):

    start_t = time.time()

    # 訓練
    model.train()
    total_loss = 0.0
    train_correct = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        predicted = torch.argmax(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        total_loss += loss.item() * labels.shape[0]
    avg_train_loss = total_loss / len(train_dataset)

    # 評価
    model.eval()
    total_loss = 0.0
    valid_correct = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            predicted = torch.argmax(outputs.data, 1)
            valid_correct += (predicted == labels).sum().item()
            total_loss += loss.item() * labels.shape[0]
    avg_valid_loss = total_loss / len(valid_dataset)

    logger.info(
        f'epoch: {epoch:2d}, '
        f'train_loss: {avg_train_loss:.6f}, '
        f'valid_loss: {avg_valid_loss:.6f}, '
        f'train_accuracy: {train_correct / len(train_dataset) * 100:6.2f}% ({train_correct:3d}/{len(train_dataset):3d}), '
        f'valid_accuracy: {valid_correct / len(valid_dataset) * 100:6.2f}% ({valid_correct:3d}/{len(valid_dataset):3d})'
    )

    if avg_valid_loss < best_valid_loss:
        dummy_input = torch.randn(1, 3, 64, 64)
        with redirect_stdout(open(os.devnull, 'w')):
            torch.onnx.export(model, dummy_input, str(f'{OUTPUT_PATH}.onnx'), verbose = False)
        best_valid_loss = avg_valid_loss

print(f'Saved {OUTPUT_PATH.stem}')