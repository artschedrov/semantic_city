import os, cv2, albumentations as A

# пути
img_dir      = 'city_dataset/image'
mask_dir     = 'city_dataset/mask'
out_img_dir  = 'city_dataset/aug/image'
out_mask_dir = 'city_dataset/aug/mask'
os.makedirs(out_img_dir,  exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

# аугментации
# transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.ShiftScaleRotate(
#         shift_limit=0.05,   # сдвиг до 5%
#         scale_limit=0.20,    # масштаб ±10%
#         rotate_limit=0,     # без вращения
#         border_mode=cv2.BORDER_REFLECT_101,
#         p=0.5
#     ),
#     A.RandomScale(scale_limit=0.2, p=0.5),
#     # A.RandomBrightnessContrast(
#     #     brightness_limit=0.2,
#     #     contrast_limit=0.2,
#     #     p=0.5
#     # ),
#     A.RandomBrightnessContrast(p=0.5),
#     A.GaussianBlur(blur_limit=(3, 7), p=0.5),
#     # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
# ])


def augmentation(count_files):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,  # сдвиг до 5%
            scale_limit=0.20,  # масштаб ±10%
            rotate_limit=0,  # без вращения
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.GridDistortion(p=0.1),
        A.OpticalDistortion(p=0.1),
        A.CLAHE(p=0.1)
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
    ])

    for idx in range(1, count_files+1):          # 001..050
        img_name  = f'img_{idx:03d}.png'
        mask_name = f'label_{idx:03d}.png'

        img_path  = os.path.join(img_dir,  img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        if not (os.path.isfile(img_path) and os.path.isfile(mask_path)):
            print(f'Пропуск: {img_name} или {mask_name} не найдены')
            continue

        image = cv2.imread(img_path)
        mask  = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # аугментация
        aug = transform(image=image, mask=mask)
        aug_img  = aug['image']
        aug_mask = aug['mask']

        # сохраняем
        cv2.imwrite(os.path.join(out_img_dir,  img_name),  aug_img)
        cv2.imwrite(os.path.join(out_mask_dir, mask_name), aug_mask)

    print('Генерация готова!')

augmentation(50)
