import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path

from geometric import PointInt, RectInt

def load_image(file_path: Path) -> np.ndarray:
    return np.array(Image.open(file_path).convert('RGB')) / 255.0
def save_image(array: np.ndarray, file_path: Path) -> None:
    Image.fromarray((array * 255 + 0.5).astype(np.uint8)).save(file_path)

def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resize_ratio = min(
        1.0, 
        size[0] / image.shape[0], 
        size[1] / image.shape[1]
    )
    return cv2.resize(image, (int(resize_ratio * image.shape[1]), int(resize_ratio * image.shape[0])))

def extract_digit_rects_2c(image: np.ndarray, draw_progress_images: bool = False):
    '''
    リザルト画面から、スコアに当たる部分を切り取る
    returns: (スコアを囲む四角形 * 5, 処理途中の画像リスト)
    '''
    
    progress_images: list[np.ndarray] | None
    if draw_progress_images:
        base_image = image.copy()
        progress_images = []
    else:
        progress_images = None

    def get_reference_position(red_area: np.ndarray):
        '''
        リザルト画面の基準となる座標を求める
        returns: (残りピクミンの座標(y, x), 残りピクミンとハイスコアの距離)
        '''

        min_count = red_area.shape[1] // 30
        
        # 赤いピクセル数をカウント
        count = red_area.sum(axis = 1)
        count = np.convolve(count, np.array([1/3, 1/3, 1/3]), mode = 'same')

        # ハイスコアの座標を求める
        highscore_top, highscore_bottom = -1, -1
        for i in reversed(range(count.shape[0])):
            if count[i] >= min_count:
                if highscore_bottom == -1:
                    highscore_bottom = i
                highscore_top = i
            elif highscore_bottom != -1:
                break
        area: np.ndarray = red_area[highscore_top : highscore_bottom + 1, :]
        first: np.ndarray = area.argmax(axis = 1)
        last: np.ndarray = area.shape[1] - 1 - area[:, ::-1].argmax(axis = 1)
        gap = int((first.shape[0] - 1) * 0.25)
        highscore_left = sorted(first)[gap]
        highscore_right = sorted(last)[len(last) - 1 - gap]

        count = red_area[:, highscore_left : highscore_right + 1].sum(axis = 1)
        count = np.convolve(count, np.array([1/3, 1/3, 1/3]), mode = 'same')

        # 残りピクミンの座標を求める
        remaining_pikmin_top, remaining_pikmin_bottom = -1, -1
        for i in reversed(range(2 * highscore_top - highscore_bottom)):
            if count[i] >= min_count:
                if remaining_pikmin_bottom == -1:
                    remaining_pikmin_bottom = i
                remaining_pikmin_top = i
            elif remaining_pikmin_bottom != -1:
                break
        highscore_pos = PointInt(
            (highscore_left + highscore_right) // 2,
            (highscore_top + highscore_bottom) // 2, 
        )
        remaining_pikmin_pos = PointInt(highscore_pos.x, (remaining_pikmin_top + remaining_pikmin_bottom) // 2)
        length = highscore_pos.y - remaining_pikmin_pos.y # ハイスコアと残りピクミンの距離
        is_wide = (highscore_right - highscore_left) / length >= 2.55 # 2.42 ~ 2.68
        return remaining_pikmin_pos, length, is_wide

    # 赤色の箇所を求める
    PIKMIN2_RESULT_RED = np.array([222 / 255, 13 / 255, 9 / 255])
    distances = np.linalg.norm(image - PIKMIN2_RESULT_RED, axis = 2)
    red_area = (distances <= 0.5).astype(float)

    # 膨張と収縮
    kernel = np.ones((3, 3), np.uint8)
    red_area = cv2.morphologyEx(red_area, cv2.MORPH_CLOSE, kernel)
    red_area = cv2.morphologyEx(red_area, cv2.MORPH_OPEN, kernel)
    if draw_progress_images:
        image = base_image.copy()
        image[red_area == 1] = np.array([0, 1, 0])
        progress_images.append(image.copy())

    # スコアの座標計算
    remaining_pikmin_pos, length, is_wide = get_reference_position(red_area)
    digits_size = int(0.5 * length) if is_wide else int(0.475 * length)
    digits_stride = 1.15 * digits_size if is_wide else digits_size
    digits_start_pos = \
        remaining_pikmin_pos - PointInt(int(2 * digits_stride + digits_size // 2), int(1.65 * digits_size))
    count = red_area.sum(axis = 1)
    hist = (np.arange(red_area.shape[1]) <= count[:, np.newaxis] - 1).astype(int)
    if draw_progress_images:
        image = base_image.copy()
        image[hist == 1] = np.array([0, 1, 0])
        image = cv2.line(
            image,
            tuple(remaining_pikmin_pos),
            tuple(remaining_pikmin_pos - PointInt.EY * length),
            (0, 0, 1),
            thickness = 2,
        )

    # スコアの座標を四角形として計算
    rects: list[RectInt] = []
    for i in range(5):
        rect_start = PointInt(digits_start_pos.x + int(i * digits_stride), digits_start_pos.y)
        rect_end = PointInt(digits_start_pos.x + digits_size + int(i * digits_stride), digits_start_pos.y + digits_size)
        rect = RectInt.from_points(rect_start, rect_end)
        rects.append(rect)
        thickness = max(
            int(((image.shape[0] * image.shape[1]) ** 0.5) / 2000),
            1,
        )
        if draw_progress_images:
            image = cv2.rectangle(
                image, 
                tuple(rect_start), 
                tuple(rect_end), 
                (1, 0, 0), 
                thickness = thickness,
            )
    if draw_progress_images:
        progress_images.append(image.copy())

    return tuple(rects), progress_images

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type = str, choices = ['train', 'new'], help='target images')
    parser.add_argument('--outdir', type = str, help='output directory path')
    args = parser.parse_args()

    if args.target == 'train':
        TRAIN_IMAGE_DIR = Path(__file__).parent / 'data/2c'
    elif args.target == 'new':
        TRAIN_IMAGE_DIR = Path(__file__).parent / 'new_images'
    else:
        assert False, args.target
    TMP_DIR = Path(args.outdir)
    DRAW_PROGRESS_IMAGES = True

    # 画像読み込み
    images: list[np.ndarray] = []
    image_paths: list[Path] = []
    print('Loading images ...')
    for image_path in tqdm(sorted(TRAIN_IMAGE_DIR.iterdir())):
        if image_path.suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.jfif']:
            image = load_image(image_path)
            images.append(image)
            image_paths.append(image_path)

    # スコア部分を抽出
    print('Extracting ...')
    for image, image_path in tqdm(zip(images, image_paths)):
        rects, progress_images = extract_digit_rects_2c(image, DRAW_PROGRESS_IMAGES)
        if DRAW_PROGRESS_IMAGES:
            image_save_path = lambda n: TMP_DIR / f'P{n:02d}_{image_path.stem}.png'
            for i, im in enumerate(progress_images):
                save_image(im, image_save_path(i))