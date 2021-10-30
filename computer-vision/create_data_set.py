import pandas as pd
from pathlib import Path
import shutil
import os
import json

image_source = 'indoor_outdoor/images'
image_dest = 'indoor_outdoor_images'
indoor_scenes = {'Bedroom', 'Bathroom', 'Classroom', 'Office', 'Living Room', 'Dining Room', 'Room'}
outdoor_scenes = {'Landscape', 'Skyscraper', 'Mountain', 'Beach', 'Ocean'}

vocab = pd.read_csv('indoor_outdoor/vocabulary.csv')
f = open('indoor_outdoor/video_category_data.json')
image_labels = json.load(f)


def create_directory(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def map_classes(metadata, indoor_classes, outdoor_classes):
    mapping_index = {'indoor': [], 'outdoor': []}
    for index, row in metadata.iterrows():
        if row['Name'] in indoor_classes:
            mapping_index['indoor'].append(row['Index'])
        if row['Name'] in outdoor_classes:
            mapping_index['outdoor'].append(row['Index'])
    return mapping_index


def map_parent_category(location_mappings):
    map_image_location = {}
    indoor_label = 0
    outdoor_label = 1
    for i in image_labels:
        if any(l in location_mappings['indoor'] for l in i['labels']):
            map_image_location[i['long_id']] = indoor_label
        if any(l in location_mappings['outdoor'] for l in i['labels']):
            map_image_location[i['long_id']] = outdoor_label
    return map_image_location


def move_images(orig_path, dest_path, image_dict):
    for file in Path(orig_path).glob("*.jpg"):
        image_id = str(file.resolve()).split('\\')[-1].replace('.jpg', '')
        if image_id in image_dict.keys():
            file_name = str(image_dict[image_id]) + '-' + image_id + '.jpg'
            final_path = os.path.join(dest_path, file_name)
            shutil.copy(file, final_path)


def main():
    create_directory(image_dest)
    location_classes = map_classes(vocab, indoor_scenes, outdoor_scenes)
    image_locations = map_parent_category(location_classes)
    move_images(image_source, image_dest, image_locations)


if __name__ == "__main__":
    main()
