
import numpy as np
import os
from collections import Counter

baseline_folder = '/data1/icpr/result/exp18/preds'
essemble_folder = [
    'model 1 result path',
    'model 2 result path',
    'model 3 result path',
    'many path ...'
]
infra_folder = '/data1/icpr/result/exp19/preds'
target_folder = '/data1/icpr/result/ess1'

# iou thres for infra and vision essemble
iou_thres = 0.45

def get_iou(bbox_a, bbox_b):
    """
    return the Iou of box A and box B
    Iou = (Area of overlap) / (Area of Union)
    """
    box_ok = lambda x : x[2] - x[0] > 0 and x[3] - x[1] > 0
    assert box_ok(bbox_a) and box_ok(bbox_b)

    ixmin = max(bbox_a[0], bbox_b[0])
    iymin = max(bbox_a[1], bbox_b[1])
    ixmax = min(bbox_a[2], bbox_b[2])
    iymax = min(bbox_a[3], bbox_b[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    
    box_dim = lambda x : ((x[2] - x[0] + 1.) * (x[3] - x[1] + 1.))
    inters = iw * ih
    uni = box_dim(bbox_a) + box_dim(bbox_b) - inters
    iou = inters / uni

    return iou

def rawtxt_read(file_path):
    with open(file_path, 'r') as f:
        ret_data = []
        for line in f:
            numbers = list(map(float, line.strip().split()))
            ret_data.append(numbers)
    return ret_data

# essemble_path中的大量模型预测结果进行wbf操作
# wbf求得的新数据在返回值中，写入文件即可
def wbf(essemble_path, file_name, base_data):
    new_data = []
    
    for path in essemble_path:
        file_path = os.path.join(path, file_name)
        pred_data = rawtxt_read(file_path)
        
        for base_bbox in base_data:
            bbox_a, conf_a, label_a = base_bbox[:4], base_bbox[4], base_bbox[5]
            if conf_a < 0.2: continue
            avg_bbox_base = [base_bbox]

            for pred_bbox in pred_data:
                bbox_b, conf_b, label_b = pred_bbox[:4], pred_bbox[4], pred_bbox[5]
                iou = get_iou(bbox_a, bbox_b)
                if iou >= 0.7:
                    avg_bbox_base.append(pred_bbox)
            
            # 分别处理 bbox, conf, 和 label
            bboxes = [bbox for bbox, conf, label in avg_bbox_base]
            confs  = [conf for bbox, conf, label in avg_bbox_base]
            labels = [label for bbox, conf, label in avg_bbox_base]

            avg_bbox_coords = [sum(xory) / len(xory) for xory in zip(*bboxes)]
            avg_conf = sum(confs) / len(confs)
            avg_label = Counter(labels).most_common(1)[0][0]
            avg_bbox = (avg_bbox_coords + [avg_conf, avg_label])
            new_data.append(avg_bbox)

    return new_data

# 超过参数 iou_thres 的所有bbox进行融合
# 两个预测结果地位等同的进行结果融合，但是bbox都用1的，label看谁更自信，conf求均值
bbox_ess_counter = 0
def equal_essemble(file_path1, conf_thres1, file_path2, conf_thres2):

    new_data = []

    data1 = rawtxt_read(file_path1)
    data2 = rawtxt_read(file_path2)

    for line1 in data1:
        bbox_a, conf_a, label_a = line1[:4], line1[4], line1[5]
        if conf_a < conf_thres1: continue
        for line2 in data2:
            bbox_b, conf_b, label_b = line2[:4], line2[4], line2[5]
            if conf_b < conf_thres2 or get_iou(bbox_a, bbox_b) < iou_thres: continue
            new_label = label_a if label_a == label_b else (label_a if conf_a > conf_b else label_b)
            new_conf  = (conf_a + conf_b) / 2
            global bbox_ess_counter
            bbox_ess_counter += 1
            new_data.append(bbox_a + [new_conf, new_label])
            break
        else: new_data.append(line1)
    
    return new_data

def folder_txt_read(folder_path):
    txt_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                txt_list.append(file_path)
                # with open(file_path, 'r') as f:
                #     txt_list.append(f)  # 将文件对象添加到列表中

    return txt_list

# 将data写到target_folder下的fname文件中
def save_data(fname, data):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"目标文件夹 {target_folder} 已创建。")
    file_path = os.path.join(target_folder, fname)
    with open(file_path, 'w') as f:
        for line in data:
            for item in line:
                f.write(str(item) + ' ')
            f.write('\n')
            


if __name__ == '__main__':
    print(f' iou_thres -> {iou_thres}')

    baseline_path_list = folder_txt_read(baseline_folder)
    infra_path_list = folder_txt_read(infra_folder)

    # print(baseline_path_list[0])
    # print(infra_path_list[0])

    counter = 0
    for b_path in baseline_path_list:
        b_name = os.path.basename(b_path)
        new_data = []
        for i_path in infra_path_list:
            i_name = os.path.basename(i_path)
            if b_name == i_name:
                # print(b_name, i_name)
                counter += 1
                new_data = equal_essemble(b_path, 0.2, i_path, 0.4)
                break
        else:
            print(f'没有匹配到相同的文件名 {b_path} and {i_path}')
            break
        save_data(b_name, new_data)
    
    print(f' {counter} 模型预测结果已融合 ✅')
    print(f' {bbox_ess_counter} bbox 已融合 ✅')
    

