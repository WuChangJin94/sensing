import json
import os

def remove_empty_segmentations_and_images(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Filter out annotations with empty segmentation
    filtered_annotations = [anno for anno in data['annotations'] if anno['segmentation']]
    
    # Get the IDs of images that have annotations
    image_ids_with_annotations = {anno['image_id'] for anno in filtered_annotations}
    
    # Filter out images that don't have annotations
    filtered_images = [img for img in data['images'] if img['id'] in image_ids_with_annotations]
    
    # Update the data with filtered annotations and images
    data['annotations'] = filtered_annotations
    data['images'] = filtered_images
    
    # Write the updated data to a new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)

input_json_path = 'water_splash_dataset_2024/annotations/instances_train2024.json'
output_json_path = 'water_splash_dataset_2024/annotations/instances_train2024.json'
remove_empty_segmentations_and_images(input_json_path, output_json_path)


input_json_path = 'water_splash_dataset_2024/annotations/instances_val2024.json'
output_json_path = 'water_splash_dataset_2024/annotations/instances_val2024.json'
remove_empty_segmentations_and_images(input_json_path, output_json_path)