import os
import glob
from pathlib import Path

import numpy as np
import cv2


IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes


def xywh2xyxy(box):
    box[:, 0] = box[:, 0] - box[:, 2] / 2
    box[:, 1] = box[:, 1] - box[:, 3] / 2
    box[:, 2] = box[:, 0] + box[:, 2]
    box[:, 3] = box[:, 1] + box[:, 3]
    return box

def bbox_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    box1 : array, shape=(4,)
        [x_min, y_min, x_max, y_max] for the first box.
    box2 : array, shape=(4,)
        [x_min, y_min, x_max, y_max] for the second box.

    Returns
    -------
    iou : float
        Intersection over Union (IoU) between box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the Union area by using formula: Union(A,B) = A + B - Inter(A,B)
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area
    return iou


def calculate_iou(box1, box2):
    ious = np.zeros((box2.shape[0],))
    for i in range(box2.shape[0]):
        ious[i] = bbox_iou(box2[i][:],box1[0][:])
    return ious


def draw_box(img, box, color):
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=2)
    return img


def main():
    img_root = '/mnt/hdd01/hyeongdo/datasets/VR-DRONE-v1.0.0.test/20221207/1Class/**/*'
    img_root = '/work/dataset/VP-SAR-v1.0.0.all/Test/**/*'
    files = []
    files.extend(sorted(glob.glob(img_root, recursive=True)))  # glob
    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

    pred_path = '/mnt/hdd01/hyeongdo/workspace/yolov9/runs/detect/motsynth_48e_1920/labels'
    pred_path = '/work/src/yolov9/runs/val/gelan-e-1280-best-20140520/labels'
    # pred_path = '/mnt/hdd01/hyeongdo/workspace/yolov9/runs/detect/motsynth_48e_1280/labels'
    # pred_path = '/mnt/hdd01/hyeongdo/workspace/yolov9/runs/detect/motsynth_48e_640/labels'
    
    save_path = '/mnt/hdd01/hyeongdo/workspace/cv-tools/data/motsynth_48e_1920'
    save_path = '/work/src/yolov9/runs/val/gelan-e-1280-best-20140520/images'

    tp_color, fn_color, fp_color = (0, 255, 0), (0, 0, 255), (255, 0, 0) # G, R, B

    os.makedirs(save_path, exist_ok=True)

    total_tp_count, total_fn_count, total_fp_count = 0, 0, 0

    for img_file in images:
        p = Path(img_file)

        # Read image
        image = cv2.imread(img_file)
        h, w = image.shape[:2] # height, width, channel

        # Read prediction
        pred_file = os.path.join(pred_path, f'{p.stem}.txt')
        try:
            with open(pred_file) as fd:
                pred = np.array(list(map(lambda x:np.array(x.strip().split(), dtype=np.float32), fd.readlines())))
                pred[:, 1:5] = xywh2xyxy(pred[:, 1:5])
                pred[:, [1, 3]] *= w
                pred[:, [2, 4]] *= h
                pred = list(pred)
        except:
            pred = []

        # Read taget(GT, Label)
        label_file = p.parent / f'{p.stem}.txt'
        try:
            with open(label_file) as fd:
                label = np.array(list(map(lambda x:np.array(x.strip().split(), dtype=np.float32), fd.readlines())))
                label[:, 1:] = xywh2xyxy(label[:, 1:])
                label[:, [1, 3]] *= w
                label[:, [2, 4]] *= h
        except:
            label = np.array([])
            print(f'label path:{label_file} (not found or no target).')

        tp_count, fn_count, fp_count, tp_box, fn_box, fp_box = _calculate_tp_fp_fn(label, pred)
        total_tp_count, total_fn_count, total_fp_count = total_tp_count + tp_count, total_fn_count + fn_count, total_fp_count + fp_count

        for box in tp_box:
            image = draw_box(image, box, tp_color)
        for box in fn_box:
            image = draw_box(image, box, fn_color)
        for box in fp_box:
            image = draw_box(image, box, fp_color)

        cv2.imwrite(f'{save_path}/{p.name}', image)
        print(f'completed to draw results on {save_path}/{p.name}(tp: {tp_count}, fp: {fp_count}, fn: {fn_count})')

def _calculate_tp_fp_fn(label, pred, iou_threshold=0.45):
    tp_count, fn_count, fp_count = 0, 0, 0
    tp_box, fn_box, fp_box = [], [], []
    
    # label은 있고 pred는 0인 경우 (all fn)
    if len(label) and len(pred) == 0:
        fn_box.extend([l[1:5] for l in label])
        fn_count = len(label)

    # pred는 있고 label은 없는 경우 (all fp)
    if len(label) == 0 and len(pred):
        fp_box.extend([l[1:5] for l in pred])
        fp_count = len(pred)

    # pred, label 모두 있는 경우
    for i in range(label.shape[0]):
        if len(pred) == 0: break
        ious = calculate_iou(label[i:i+1, 1:], np.array(pred)[:, 1:5]) # [[ious]]
        ious_argsort = ious.argsort()[::-1] # decending sort
        missing = True

        for j in ious_argsort:
            if ious[j] < iou_threshold: break
            if label[i, 0] == pred[j][0]:
                tp_box.append(pred[j][1:5])
                
                missing = False
                tp_count += 1

                pred.pop(j)
                break
        
        if missing:
            fn_box.append(label[i][1:5])
            fn_count += 1

    if len(pred):
        for j in range(len(pred)):
            fp_box.append(pred[j][1:5])
            fp_count += 1  

    return tp_count, fn_count, fp_count, tp_box, fn_box, fp_box


if __name__ == '__main__':
    main()
