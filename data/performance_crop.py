import imageio
import tqdm
import cv2
from multiprocessing import Pool
import os

# RESET: rm -rf ./PBN_crops/

size = 256

try:
    os.makedirs('./data/art_cropped_train/data/')
except FileExistsError:
    pass

def scale(file):
    newfile = file.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    newpath = './data/art_cropped_train/data/' + newfile

    if os.path.exists(newpath + '.jpg'):
        return True

    try:
        image = imageio.imread('./data/train/' + file)
    except Exception as e:
        print('imread error: ' + file)
        print(e)
        return False

    # If it's not a grayscale image and it is a rgb*a* image, remove the alpha channel
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    w, h = image.shape[1], image.shape[0]
    ratio = w / h

    scale_factor = size / w if w < h else size / h

    image = cv2.resize(image, dsize=(round(w * scale_factor), round(h * scale_factor)), interpolation=cv2.INTER_LINEAR)

    w, h = image.shape[1], image.shape[0]
    if min(w, h) != size:
        raise RuntimeError('Error')

    imageio.imwrite(newpath + '.jpg', image)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


files_chunks = list(chunks(os.listdir('./data/train/'), 1000))

for files_chunk in tqdm.tqdm(files_chunks):
    pool = Pool(24)
    results = pool.map(scale, files_chunk)
    pool.close()
    pool.join()
