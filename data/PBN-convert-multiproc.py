import imageio
import tqdm
import cv2
from multiprocessing import Pool
import os


# RESET: rm -rf ./PBN_crops/

size = 256

try:
    os.mkdir('./PBN_crops/')
except FileExistsError:
    pass


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def crop_and_resize(file):
    newfile = file.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    newpath = './PBN_crops/' + newfile

    if os.path.exists(newpath + '.jpg'):
        return True

    try:
        image = imageio.imread('./train/' + file)
    except Exception as e:
        print('imread error: ' + file)
        print(e)
        return False

    """h, w = image.shape[0], image.shape[1]
    if not (65 / 100 < (h / w) < 100 / 65):
        return False

    s = min(h, w)
    if w > h:
        try:
            image0 = image[:, w//2 - s//2:w//2 + s//2]
        except Exception as e:
            print('crop error: ' + file)
            print(e)
            return False
    elif h == w:
        image0 = image
    else:
        try:
            image0 = image[h // 2 - s // 2:h // 2 + s // 2, :]
        except Exception as e:
            print('crop error: ' + file)
            print(e)
            return False

    try:
        image0 = cv2.resize(image0, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print('resize error: ' + file)
        print(e)
        return False

    try:
        imageio.imwrite(newpath, image0)
    except Exception as e:
        print('imwrite error: ' + file)
        print(e)
        return False"""

    # If it's not a grayscale image and it is a rgb*a* image, remove the alpha channel
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    w, h = image.shape[1], image.shape[0]
    ratio = w / h

    scale_factor = size / w if w < h else size / h

    image = cv2.resize(image, dsize=(round(w * scale_factor), round(h * scale_factor)), interpolation=cv2.INTER_AREA)

    w, h = image.shape[1], image.shape[0]

    save_shifted = False
    save_shifted1 = False

    if w > h:
        cropped = image[0:size, w // 2 - size // 2:w // 2 + size // 2]
        if w * 0.35 > size // 2 and w - w * 0.65 > size // 2:
            cropped_shift_left = image[0:size, int(w * 0.35) - size // 2:int(w * 0.35) + size // 2]
            cropped_shift_right = image[0:size, int(w * 0.65) - size // 2:int(w * 0.65) + size // 2]
            save_shifted = True
        if w * 0.45 > size // 2 and w - w * 0.55 > size // 2:
            cropped_shift1_left = image[0:size, int(w * 0.45) - size // 2:int(w * 0.45) + size // 2]
            cropped_shift1_right = image[0:size, int(w * 0.55) - size // 2:int(w * 0.55) + size // 2]
            save_shifted1 = True
    else:
        cropped = image[h // 2 - size // 2:h // 2 + size // 2, 0:size]
        if h * 0.35 > size // 2 and h - h * 0.65 > size // 2:
            cropped_shift_left = image[int(h * 0.35) - size // 2:int(h * 0.35) + size // 2, 0:size]
            cropped_shift_right = image[int(h * 0.65) - size // 2:int(h * 0.65) + size // 2, 0:size]
            save_shifted = True
        if h * 0.45 > size // 2 and h - h * 0.55 > size // 2:
            cropped_shift1_left = image[int(h * 0.45) - size // 2:int(h * 0.45) + size // 2, 0:size]
            cropped_shift1_right = image[int(h * 0.55) - size // 2:int(h * 0.55) + size // 2, 0:size]
            save_shifted1 = True


    imageio.imwrite(newpath + '.jpg', cropped)
    if save_shifted:
        imageio.imwrite(newpath + 'LL.jpg', cropped_shift_left)
        imageio.imwrite(newpath + 'RR.jpg', cropped_shift_right)
    if save_shifted1:
        imageio.imwrite(newpath + 'L.jpg', cropped_shift1_left)
        imageio.imwrite(newpath + 'R.jpg', cropped_shift1_right)

    return True


files_chunks = list(chunks(os.listdir('./train/'), 1000))

for files_chunk in tqdm.tqdm(files_chunks):
    pool = Pool(24)
    results = pool.map(crop_and_resize, files_chunk)
    pool.close()
    pool.join()

# os.system('find ./PBN_crops/ -size  0 -print0 |xargs -0 rm --')
