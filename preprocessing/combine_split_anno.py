import json
import os
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_data(directory, json_file_name):
    json_path = os.path.join(directory, json_file_name)
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def save_data(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def copy_images(src_dir, dest_dir, images):
    os.makedirs(dest_dir, exist_ok=True)
    for image in images:
        image_name = image['file_name']
        src_path = os.path.join(src_dir, image_name)
        dest_path = os.path.join(dest_dir, image_name)
        os.system(f'cp {src_path} {dest_path}')

def main(args):
    output_dir = args.output_dir
    train_images = []
    val_images = []
    train_annotations = []
    val_annotations = []

    os.makedirs(os.path.join(output_dir, 'train2024'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val2024'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    # Process each directory and subdirectory
    for base_dir in tqdm(args.dirs, desc='Processing directories', leave=False):
        for folder in tqdm(args.sub_dirs, desc='Processing subdirectories', leave=False):
            data = load_data(os.path.join(base_dir, folder, 'annotations'), f'{folder}.json')
            
            # Ensure annotations are paired with images
            image_id_to_annotations = {annotation['image_id']: annotation for annotation in data['annotations']}
            paired_images = []
            paired_annotations = []
            for image in data['images']:
                image_id = image['id']
                if image_id in image_id_to_annotations:
                    paired_images.append(image)
                    paired_annotations.append(image_id_to_annotations[image_id])

            images_train, images_val, annotations_train, annotations_val = train_test_split(
                paired_images, paired_annotations, train_size=0.8, random_state=42)

            # Copy images to respective directories
            src_image_dir = os.path.join(base_dir, folder, 'images')
            copy_images(src_image_dir, os.path.join(output_dir, 'train2024'), images_train)
            copy_images(src_image_dir, os.path.join(output_dir, 'images'), images_train)
            copy_images(src_image_dir, os.path.join(output_dir, 'val2024'), images_val)
            copy_images(src_image_dir, os.path.join(output_dir, 'images'), images_val)

            # Check images are all copied, if not then copy again
            images_num = len(os.listdir(os.path.join(base_dir, folder, 'images')))
            for i in range(images_num):
                image_name = f"{folder}_{i+1}.jpg"
                src_path = os.path.join(base_dir, folder, 'images', image_name)
                # train or val
                if image_name in [image['file_name'] for image in images_train]:
                    dest_path = os.path.join(output_dir, 'train2024', image_name)
                elif image_name in [image['file_name'] for image in images_val]:
                    dest_path = os.path.join(output_dir, 'val2024', image_name)
                else:
                    raise ValueError(f"Image {image_name} not found in train or val images")
                if not os.path.exists(dest_path):
                    print(f"Missing image: {src_path}, copy to {dest_path}")
                    os.system(f'cp {src_path} {dest_path}')

            # Accumulate data for combined JSON
            train_images.extend(images_train)
            val_images.extend(images_val)
            train_annotations.extend(annotations_train)
            val_annotations.extend(annotations_val)

            # stats
            print(f"Directory: {base_dir}/{folder}")
            print(f"Train images: {len(images_train)}")
            print(f"Val images: {len(images_val)}")
            print(f"Train annotations: {len(annotations_train)}")
            print(f"Val annotations: {len(annotations_val)}")
            print()

            with open(os.path.join(output_dir, 'stats.txt'), 'a') as f:
                f.write(f"Directory: {base_dir}/{folder}\n")
                f.write(f"Train images: {len(images_train)}\n")
                f.write(f"Val images: {len(images_val)}\n")
                f.write(f"Train annotations: {len(annotations_train)}\n")
                f.write(f"Val annotations: {len(annotations_val)}\n")
                f.write('\n')

    # Save combined JSON files
    train_data = {'images': train_images, 'annotations': train_annotations, 'categories': data['categories']}
    val_data = {'images': val_images, 'annotations': val_annotations, 'categories': data['categories']}
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    save_data(train_data, os.path.join(output_dir, 'annotations', 'instances_train2024.json'))
    save_data(val_data, os.path.join(output_dir, 'annotations', 'instances_val2024.json'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine and split annotations and images from multiple directories")
    parser.add_argument('-d', '--dirs', nargs='+', default=['labels_output_0522'], help='List of base directories to process')
    parser.add_argument('-sd', '--sub_dirs', nargs='+', default=['aerator_flow_img', 'aeratorcompare10M_flow_img', 'aeratorcompare20M_flow_img', 'aeratorcompare30M_flow_img'], help='List of subdirectories in each base directory')
    parser.add_argument('-o', '--output_dir', type=str, default='water_splash_dataset_2024', help='Output directory for combined and split data')
    args = parser.parse_args()
    main(args)
