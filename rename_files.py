import os
from aug_images import augmentation


img_dir = 'city_dataset/aug/image'
label_dir = 'city_dataset/aug/mask'
dataset_img_dir = 'city_dataset/augment_500/image'
dataset_label_dir = 'city_dataset/augment_500/mask'
scale_int = 50

import os
import shutil


def rename_files(scale_num, type_file, source_dir):
    for idx in range(1, 51):  # от 1 до 50
        old_name = f'{type_file}_{idx:03d}.png'
        new_name = f'{type_file}_{idx + scale_num:03d}.png'

        old_path = os.path.join(source_dir, old_name)
        new_path = os.path.join(source_dir, new_name)

        if os.path.isfile(old_path):  # Проверяем, существует ли старый файл
            os.rename(old_path, new_path)  # Переименовываем
            print(f'Переименован: {old_name} -> {new_name}')
        else:
            print(f'Файл не найден: {old_name}')

    print('Переименовано!')


def move_files(source_folder, destination_folder):
    # Проверяем, существует ли исходная папка
    if not os.path.exists(source_folder):
        print(f"Исходная папка '{source_folder}' не существует.")
        return

    # Создаем целевую папку, если она не существует
    os.makedirs(destination_folder, exist_ok=True)

    # Перемещаем все файлы из исходной папки в целевую
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)

        # Проверяем, является ли элемент файлом
        if os.path.isfile(source_file):
            shutil.move(source_file, destination_file)
            print(f"Файл '{filename}' перемещен в '{destination_folder}'.")

for x in range(1, 10):
    augmentation(50)
    rename_files(scale_int, 'img', img_dir)
    rename_files(scale_int, 'label', label_dir)

    move_files(img_dir, dataset_img_dir)
    move_files(label_dir, dataset_label_dir)
    scale_int +=50
# rename_files(scale_int, 'img', img_dir)
# rename_files(scale_int, 'label', label_dir)

