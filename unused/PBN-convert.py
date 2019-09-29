import imageio
import os
import tqdm
import cv2
from multiprocessing.dummy import Pool

try:
    os.mkdir('./PBN_crops/')
except FileExistsError:
    pass

total = 0
errors = 0
for file in tqdm.tqdm(os.listdir('./train/')):
    try:
        image = imageio.imread('./train/' + file)
    except:
        errors += 1
        continue

    newfile = file.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')

    h, w = image.shape[0], image.shape[1]
    if not (65 / 100 < (h / w) < 100 / 65):
        continue
    total += 1

    s = min(h, w)
    if w > h:
        try:
            image0 = image[:, w//2 - s//2:w//2 + s//2]
            image0 = cv2.resize(image0, dsize=(400, 400), interpolation=cv2.INTER_LINEAR)
            imageio.imwrite('./PBN_crops/' + newfile + '.jpg', image0)
        except: pass
    elif h == w:
        pass
    else:

        try:
            image0 = image[h // 2 - s // 2:h // 2 + s // 2, :]
            image0 = cv2.resize(image0, dsize=(400, 400), interpolation=cv2.INTER_LINEAR)
            imageio.imwrite('./PBN_crops/' + newfile + '.jpg', image0)
        except: pass




print(total)
print(errors)
