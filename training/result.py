import torch
import torchvision.transforms.functional as F
from PIL import Image
from pathlib import Path

from number_extractor import extract_digit_rects_2c

class Pikmin2cResult:
    LABELS = [str(i) for i in range(10)] + [' ']
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.image = F.to_tensor(Image.open(image_path).convert('RGB'))
        tokens = image_path.stem.split('-')
        assert len(tokens) == 3
        self.stage_id = int(tokens[0])
        self.player_name = tokens[1]
        self.score = int(tokens[2])
        score_str = (' ' * 5 + str(self.score))[-5:]
        self.labels = torch.tensor([Pikmin2cResult.LABELS.index(c) for c in score_str] + [self.stage_id - 1])
    @staticmethod
    def create_data_pairs(results: list['Pikmin2cResult']):
        images: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for result in results:
            image = result.image
            rects, _ = extract_digit_rects_2c(image.permute(1, 2, 0).numpy())
            for rect, label in zip(rects, result.labels):
                sub_image = image[:, rect.y : rect.y + rect.h + 1, rect.x : rect.x + rect.w + 1]
                sub_image = F.resize(sub_image, (64, 64))
                sub_image = sub_image.to(torch.float32)
                images.append(sub_image)
                labels.append(label)
        return torch.stack(images), torch.stack(labels)