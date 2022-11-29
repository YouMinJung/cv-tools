def iou(box_a, box_b):
    # box_a: [min_x, min_y, max_x, max_y]
    # box_b: [min_x, min_y, max_x, max_y]
    min_x = max(box_a[0], box_b[0])
    min_y = max(box_a[1], box_b[1])
    max_x = min(box_a[2], box_b[2])
    max_y = min(box_a[3], box_b[3])

    inter_area = max(0, max_x - min_x + 1) * max(0, max_y - min_y + 1)
    a_box_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    b_box_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = inter_area / float(a_box_area + b_box_area - inter_area)

    return iou


if __name__ == '__main__':
    pass