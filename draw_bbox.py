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


def iou(box1, box2):
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)

    xa = np.maximum(x11, np.transpose(x21))
    xb = np.maximum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.maximum(y12, np.transpose(y22))

    area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))

    area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    area_union = area_1 + np.transpose(area_2) - area_inter

    iou = area_inter / area_union
    return iou


def draw_box(img, box, color):
    # box: [y1 x1 y2 x2]
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=2)
    # print((int(box[0]), int(box[1])), (int(box[2]), int(box[3])), img.shape)
    # cv2.rectangle(img, (138, 1203), (143, 1226), (0,0, 255), thickness=2)
    # cv2.rectangle(img, (100, 100), (143, 1220), (0,0, 255), thickness=2)
    # test = '/mnt/hdd01/hyeongdo/workspace/cv-tools/data/test.jpg'
    # cv2.imwrite(test, img)
    return img


def _calculate_tp_fp_fn(label, pred, iou_threshold=0.45):
    tp_count, fn_count, fp_count = 0, 0, 0
    tp_box, fn_box, fp_box = [], [], []
    # label_id, pred_id = list(range(label.shape[0])), [] if len(pred)==0 else list(range(len(pred)))
    
    # label은 있고 pred는 0인 경우 (all fn)
    if len(label) and len(pred) == 0:
        # return tp_count, fn_count, fp_count, tp_box, fn_box, fp_box
        fn_box.extend([l[1:5] for l in label])
        fn_count = len(label)

    # pred는 있고 label은 없는 경우 (all fp)
    if len(label) == 0 and len(pred):
        # return tp_count, fn_count, fp_count, tp_box, fn_box, fp_box
        fp_box.extend([l[1:5] for l in pred])
        fp_count = len(pred)

    # pred, label 모두 있는 경우
    for i in range(label.shape[0]):
        if len(pred) == 0: break
        ious = iou(label[i:i+1, 1:], np.array(pred)[:, 1:5])[0] # [[ious]]
        ious_argsort = ious.argsort()[::-1] # decending sort
        missing = True

        for j in ious_argsort:
            if ious[j] < iou_threshold: break
            if label[i, 0] == pred[j][0]:
                # image = draw_box(image, pred[j][1:5], tp_color)
                tp_box.append(pred[j][1:5])
                
                missing = False
                tp_count += 1

                pred.pop(j)
                break
        
        if missing:
            # image = draw_box(image, label[i][1:5], fn_color)
            fn_box.append(label[i][1:5])
            fn_count += 1

    if len(pred):
        for j in range(len(pred)):
            # image = draw_box(image, pred[j][1:5], fp_color)
            fp_box.append(pred[j][1:5])
            fp_count += 1  

    return tp_count, fn_count, fp_count, tp_box, fn_box, fp_box


if __name__ == '__main__':

    img_root = '/mnt/hdd01/hyeongdo/datasets/VR-DRONE-v1.0.0.test/20221207/1Class/**/*'
    files = []
    files.extend(sorted(glob.glob(img_root, recursive=True)))  # glob
    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    postfix = 'jpg'
    img_path = 'image'
    label_path = 'label'
    predict_path = 'predict'
    pred_path = '/mnt/hdd01/hyeongdo/workspace/yolov9/runs/detect/motsynth_48e_1920/labels'
    save_path = '/mnt/hdd01/hyeongdo/workspace/cv-tools/data'
    classes = ['p', 'm']
    tp_color, fn_color, fp_color = (0, 255, 0), (0, 0, 255), (255, 0, 0) # G, R, B

    iou_threshold = 0.45
    os.makedirs(save_path, exist_ok=True)

    total_tp_count, total_fn_count, total_fp_count = 0, 0, 0

    for img_file in images:
        p = Path(img_file)
        basename = p.stem

        # Read image
        image = cv2.imread(img_file)
        h, w = image.shape[:2] # height, width, channel
        # print(image.shape)

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

        # label path에 있는 모든 label을 읽는다
        # pred path에 있는 label을 읽는다.
        # 그러면 어떤 구조에서 데이터를 읽어 올 것인가?

        tp_count, fn_count, fp_count, tp_box, fn_box, fp_box = _calculate_tp_fp_fn(label, pred)
        total_tp_count, total_fn_count, total_fp_count = total_tp_count + tp_count, total_fn_count + fn_count, total_fp_count + fp_count
        # print(f'tp: {total_tp_count}, fn: {total_fn_count}, fp: {total_fp_count}')
        # img = image.copy()
        # print(f'tp: {tp_box}',f'fn: {fn_box}\n fp: {fp_box}')
        for box in tp_box:
            image = draw_box(image, box, tp_color)
        for box in fn_box:
            image = draw_box(image, box, fn_color)
        for box in fp_box:
            image = draw_box(image, box, fp_color)

        cv2.imwrite(f'{save_path}/{p.name}', image)
        print(f'completed to draw results on {save_path}/{p.name}(tp: {tp_count}, fp: {fp_count}, fn: {fn_count})')
