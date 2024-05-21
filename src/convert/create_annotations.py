import os

def create_image_annotation(file_path, width, height, image_id, dataset_root):
    file_path = str(file_path)

    if not dataset_root.endswith(os.sep):
        dataset_root += os.sep

    file_path = file_path.replace(dataset_root, "")
    print(file_path)
    image_annotation = {
        'file_name': file_path,
        'height': height,
        'width': width,
        'id': image_id
    }
    return image_annotation


def create_annotation_from_yolo_format(
        min_x, min_y, width, height, image_id, category_id, annotation_id, segmentation=True
):
    bbox = (float(min_x), float(min_y), float(width), float(height))
    area = width * height
    max_x = min_x + width
    max_y = min_y + height
    if segmentation:
        seg = [[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]]
    else:
        seg = []

    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "category_id": category_id,
        "segmentation": seg,
    }

    return annotation


def create_annotation_from_yolo_results_format(
    min_x, min_y, width, height, image_id, category_id, conf
):
    bbox = (float(min_x), float(min_y), float(width), float(height))

    annotation = [{
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "score": conf
    }]

    return annotation

coco_format = {"images": [{}], "categories":[], "annotations": [{}]}
