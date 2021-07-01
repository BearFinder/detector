from typing import List, Tuple
from datetime import datetime
import xml

import cv2 as cv
import numpy as np
from PIL import Image as im
from scipy import ndimage



def preprocess_img(img_path: str, coords: List[List[int]], size=112) -> None:
    """
    img_path (str): Path to the image.
    coords (List[List[int]]): Coordinates of the bears (ex.: [[x1, y1, w1, h1], [x2, y2, w2, h2]).
    """
    EMPTY_IMAGES: int = 0
    BEAR_IMAGES: int = 0
    img = cv.imread(img_path)
    mask = np.zeros_like(img[:, :, 0:1])
    
    for i, (x, y, w, h) in enumerate(coords):
        print(f'Processing bear {i + 1}/{len(coords)}...')
        mask[y : y+h, x : x+w] = 1
        composite = np.concatenate([img, mask], axis=2)
        for angle in range(0, 360, 15):
            print(f'\t* rotating by {angle}º/360º')
            composite_rotated = ndimage.rotate(composite, angle)
            mask_rotated = composite_rotated[:, :, 3]
            max_x = mask_rotated.max(axis=0).nonzero()[0].max()
            min_x = mask_rotated.max(axis=0).nonzero()[0].min()
            max_y = mask_rotated.max(axis=1).nonzero()[0].max()
            min_y = mask_rotated.max(axis=1).nonzero()[0].min()
            center = ((max_x + min_x) // 2, (max_y + min_y) // 2)
            
            BBOXs: List[List[Tuple[int]]] = [
                        [(center[0] - size // 3, (center[0] - size // 3) + size),
                        (center[1] - size // 3, (center[1] - size // 3) + size)],
                        
                        [(center[0] - size // 3, (center[0] - size // 3) + size),
                        (center[1] - 2 * size // 3, (center[1] - 2 * size // 3) + size)],
                        
                        [(center[0] - 2 * size // 3, (center[0] - 2 * size // 3) + size),
                        (center[1] - size // 3, (center[1] - size // 3) + size)],
                        
                        [(center[0] - 2 * size // 3, (center[0] - 2 * size // 3) + size),
                        (center[1] - 2 * size // 3, (center[1] - 2 * size // 3) + size)],
                        
                        [(center[0] - size // 2, (center[0] - size // 2) + size),
                        (center[1] - size // 2, (center[1] - size // 2) + size)]
            ]
            
            for i, bbox in enumerate(BBOXs):
                print(f'\t\t- B-box #{i + 1}')
                BEAR_IMAGES += 1
                cv.imwrite(
                    'train/with_bears/' + '_'.join(str(datetime.now()).split()) + '.png',
                    composite_rotated[bbox[1][0]:bbox[1][1], bbox[0][0]:bbox[0][1], 0:3]
                )
        
        cur_x, cur_y = 0, 0
        while cur_x < img.shape[1] and cur_y < img.shape[0]:
            if x < cur_x < x + w and y < cur_y < y + h:
                pass
            else:
                EMPTY_IMAGES += 1
                print(f'\t\t\tEmpty image #{EMPTY_IMAGES}')
                cv.imwrite(
                    'train/empty/' + '_'.join(str(datetime.now()).split()) + '.png',
                    img[cur_y : cur_y+size, cur_x : cur_x+size, 0:3]
                )
            cur_x += size
            cur_y += size
            
            
    print(f'\n\nTOTAL STATISTICS:\n\tImages with bears: {BEAR_IMAGES}\n\tEmpty images: {EMPTY_IMAGES}')



def process_empty_images(img_path: str, EMPTY_PER_IMAGE=93) -> None:
    """
    img_path (str): Path to the image.
    EMPTY_PER_IMAGE (int): Amout of the frames from the current image.
    """
    SIZE = 112
    img = cv.imread(img_path)
    cnt = 0
    run = True
    print(f'Processing image {img_path}...')
    for x in range(0, img.shape[1], SIZE):
        for y in range(0, img.shape[0], SIZE):
            if cnt > EMPTY_PER_IMAGE - 1:
                run = False
                break
            else:
                cnt += 1
                print(f'\t- Write #{cnt}')
                cv.imwrite(
                    'train/empty/' + '_'.join(str(datetime.now()).split()) + '.png',
                    img[y : y+SIZE, x : x+SIZE, 0:3]
                )
        if not run:
            break
