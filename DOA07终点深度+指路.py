from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import os
from data import cfg, set_cfg, set_dataset
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path   #pathlib包提供Path函数用于访问文件路径
from collections import OrderedDict
#OrderedDict是Python中collections模块中的一种字典，它可以按照元素添加的顺序来存储键值对，保证了元素的顺序性。
#与Python中普通字典不同的是，OrderedDict中的元素顺序是有序的，而且支持所有字典的操作。
#OrderedDict类的优点是能够维护元素的有序性，缺点是存储元素时需要使用双向链表，因此需要比字典类占用更多的内存空间
#字典所拥有的方法 OrderedDict 也同样
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import pyrealsense2 as rs
import datetime
from PathPlanning.Search_2D.BAStar_g_h_turn import BidirectionalAStar, MapProcessor
# from PathPlanning.Search_2D.D_star_Lite import DStar, MapProcessor
from PathPlanning.Search_2D import plotting
import pyttsx3
#################################################20241218播报模块导入##
# 初始化语音引擎
engine = pyttsx3.init()
# 设置语速（可选）
rate = engine.getProperty('rate')
engine.setProperty('rate', rate +10)

# 设置音量（可选）
volume = engine.getProperty('volume')
engine.setProperty('volume', volume + 0.25)

# 设置语音（可选）
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[12].id)  # 选择不同的语音,12号是中文
last_message = ""
def speak_direction(direction, position,dis):
    global last_message  # 声明 last_message 为全局变量
    # 构造当前要播报的信息
    # # 将 dis 拆分为整数部分和小数部分
    
    integer_dis = int(dis)  # 整数部分
    decimal_dis = int((dis - integer_dis) * 10)  # 小数部分（乘以10并取整）

    # 构造当前要播报的信息
    #current_message = f"终点在{direction},{position},{integer_dis}米{decimal_dis}处"
    current_message = f"终点在{direction},{position},{integer_dis}米"
    #print(current_message)
    # 检查当前消息是否与上一个消息一致，消息变了才播报，否则不播报
    if current_message != last_message:
        update_count = 0  # 如果消息不一致，重置计数器
        update_count += 1
        if update_count >= 2 or current_message != last_message:
            engine.say(current_message)
            engine.runAndWait()  # 等待播报结束
            print("···播报···")
        last_message = current_message  # 更新上一次的消息

def speak_out(message):
    
    # 构造当前要播报的信息
    current_message = message
    engine.say(current_message)
    engine.runAndWait()  # 等待播报结束
##############################################################20241218

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='/home/nvidia/YOLACT2plus_Blind_Nav_System/weights/yolact_base_399_100000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=15, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections '
                             'into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for '
                             'usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does '
                             'not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the '
                             'format input->output.')
    parser.add_argument('--video', default=1, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=15, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0.5, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently '
                             'only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: '
                             'coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works '
                             'for --display and --benchmark.')
    parser.add_argument('--display_fps', default=True, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                        shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                        display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True

    if args.seed is not None:
        random.seed(args.seed)


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
#创建了一个阈值列表，从50%到100%，步长为5%，用于目标检测或图像分割任务中，以确定预测的边界框与真实值之间的重叠程度
coco_cats = {}  # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})


def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    global start_in_obstacle, PathPlan_map,PathPlanStop_map
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                        crop_masks=args.crop,
                        score_threshold=args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].detach().cpu().numpy() for x in t[:3]]


    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color
    #上面这些都是识别部分，别动
    # *****************************************************
    def get_3d_camera_coordinate(pixel, aligned_depth_frame, depth_intrin):
        dis=''
        x = pixel[0]
        y = pixel[1]
        if x <= 0 or y <= 0 or x >= aligned_depth_frame.get_width() or y >= aligned_depth_frame.get_height():
            x=10
            y=10
        else:
            print('x,y:',x,y)
        dis = aligned_depth_frame.get_distance(x, y)        # 获取该像素点对应的深度
        dis = aligned_depth_frame.get_distance(x, y)        # 获取该像素点对应的深度
        # print ('depth: ',dis)       # 深度单位是m
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, pixel, dis)
        # print ('camera_coordinate: ',camera_coordinate)
        if dis ==None:
            dis=''
            return dis, camera_coordinate
        else:
            return dis, camera_coordinate
    #*上面是读取指定点的坐标和深度


    #
    def abstract_corners(path):
        corners = []  # 定义拐点列表
        for i in range(1, len(path) - 1):
            prev_point = path[i - 1]
            curr_point = path[i]
            next_point = path[i + 1]
            #若1和3的横总坐标都不相同，则2为拐点，放入拐点列表
            if prev_point[0] != next_point[0] and prev_point[1] != next_point[1]:
                corners.append(curr_point)  # 将当前点加入拐点列表
        # 将corners列表转换为索引从1开始的字典
        corners_dict = {index + 1: value for index, value in enumerate(corners)}    
        print("原路径中的拐点：", corners_dict)
        return corners_dict
    
    #########################################################################
    time1= datetime.datetime.now()
    print('开始制作地图')
    map0 = np.zeros((int(h), int(w), 3))
    map1 = torch.zeros(int(h), int(w), 1, dtype=torch.long)
    start_node = None  # 初始化 start_node

    map_goal_0 = np.zeros((int(h), int(w), 3))
    map_goal_1 = torch.zeros(int(h), int(w), 1, dtype=torch.long)

    map_goal_bigger_0 = np.zeros((int(h), int(w), 3))
    map_goal_bigger_1 = torch.zeros(int(h), int(w), 1, dtype=torch.long)

    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        #num_dets_to_consider目标识别到的结果的个数
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        #
        # **************************************************
        for ID in range(len(classes)):
            if classes[ID] == 1 :  # ground对应 ID=1
                map1 = torch.add(map1, masks[ID, :])

            if classes[ID] == 0:  # people对应 ID=0、
                
                map1 = torch.add(map1, masks[ID, :])
            
                map_goal_1 = torch.add(map_goal_1, masks[ID, :])

        map0 = (map1 * 255).byte().cpu().numpy()#单通道灰度图像
        ######################################
        kernel = np.ones((15, 15), np.uint8)  # 15X15的结构元素
        dilated_map0 = cv2.dilate(map0, kernel, iterations=1)#膨胀
        #dilated_map0 = np.concatenate((dilated_map0, dilated_map0, dilated_map0), axis=-1)#灰度-->三通道
        dilated_map0  = np.stack((dilated_map0,) * 3, axis=-1)
        
        ######################################
        
        map_processor = MapProcessor(dilated_map0, 0.3)#膨胀后的图像作整体压缩
        downsmaple_map = map_processor.downsample_map

        map_goal_0 = (map_goal_1 * 255).byte().cpu().numpy()
        map_goal_0 = np.concatenate((map_goal_0, map_goal_0, map_goal_0), axis=-1)

        
        #单通道灰度图像变成三通道，通过重复 map_goal_0 三次，可以确保在 R、G、B 三个通道中拥有相同的灰度值，这样可以保持图像的原始亮度和对比度
        #拼接操作: np.concatenate 函数通过指定 axis=-1 将数组沿最后一个维度（通道维度）进行拼接。假设 map_goal_0 的形状是 (height, width), 经过拼接后，新数组的形状将变为 (height, width, 3)，表示包含三个相同的通道。
        map_goal_processor = MapProcessor(map_goal_0, 0.3)#膨胀后的目标图像作整体压缩
        time2= datetime.datetime.now()
        print('制作地图耗时：%d ms' % ((time2 - time1).seconds * 1000 + (time2 - time1).microseconds / 1000))
        #确定起点
        start_node = (int(map_processor.map_h-2), int(map_processor.map_w / 2))
        print("\n起点：", start_node)

        start_in_obstacle = (downsmaple_map[start_node[0], start_node[1]] == 0).any()
        #压缩地图上起点的位置是否==0，0表示障碍物，
        #any() 函数用于检查提取的值中是否存在任何一个是 True，如果有，则意味着起点在障碍物中
        if start_in_obstacle:
            print("起点在障碍物中\n")
            # 起点在障碍物中，停止规划，在地图上显示静止通行的标志。
            PathPlanStop_map = map_processor.circle_line()       
        else:
            # 找终点
            
            if np.mean(map_goal_0) > 0:
                print("map_goal_0 包含非零值，说明识别到人了")
                goal_node = map_processor.find_farmost_node()
                #######压缩地图（0.3倍压缩）终点，还原到原图像上的终点，取整  
                x, y = goal_node[0]
                a = int(x / 0.3)
                b = int(y / 0.3)
                goal_in_orign_frame = (a,b)
                ###########获取深度
                jvli_float,xyz=get_3d_camera_coordinate(goal_in_orign_frame, aligned_depth_frame, depth_intrin)
                jvli_rounded = round(jvli_float, 1)
                print("到目标还有", jvli_float, "米")
                print("目标在相机坐标系下的三维坐标为",xyz)
                
            else:        
                goal_node=map_processor.find_farmost_node()
                jvli_rounded=0

            print("\n主函数得到终点：", goal_node[0])
            #终点在相机坐标系下的坐标
            
            #终点与相机的相对距离
            #############################################################20241115
            #计算终点的偏角用于播报
            front_pixel=(int(10), int(map_processor.map_w / 2)) ## 中轴像素点
            start_node_report = np.array(start_node)  # 起点
            front_pixel_report = np.array(front_pixel)  # 中轴像素点
            goal_node_report = np.array(goal_node[0])  # 终点
            # 计算方向向量
            vector_front = front_pixel_report - start_node_report  # 从起点到前像素的向量
            vector_goal = goal_node_report - start_node_report  # 从起点到目标的向量
            # 计算向量的夹角
            # 计算向量的模（长度）
            norm_front = np.linalg.norm(vector_front)
            norm_goal = np.linalg.norm(vector_goal)
            # 计算向量的点积
            dot_product = np.dot(vector_front, vector_goal)
            # 计算夹角（使用 arccos）
            cos_angle = dot_product / (norm_front * norm_goal)
            angle_radians = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 使用 clip 确保值在 -1 到 1 之间
            angle_degrees = np.degrees(angle_radians)  # 转换为度


            # 计算叉积
            cross_product = vector_front[0] * vector_goal[1] - vector_front[1] * vector_goal[0]
            direction="您"
            position="正前方"

            # 判断终点的位置
            if cross_product > 0:
                position = "左手边"
                if angle_degrees < 10:
                    direction = "12点钟方向"
                elif 10 <= angle_degrees < 45:
                    direction = "11点钟方向"
                else:
                    direction = "10点钟方向"
            elif cross_product < 0:
                position = "右手边"
                if angle_degrees < 10:
                    direction = "12点钟方向"
                elif 10 <= angle_degrees < 45:
                    direction = "1点钟方向"
                else:
                    direction = "2点钟方向"
            else:
                position = "正前方"

            
            current_message = f"终点在{direction},{position}，{jvli_rounded}米处"
            print(current_message)
            speak_direction(direction, position,jvli_rounded)

        #############################################################20241115
            PathPlan_map=None
            time3=datetime.datetime.now()
            print('具备起点终点')
            bastar = BidirectionalAStar(start_node, goal_node[0], "euclidean", downsmaple_map, min_d=1)
            path, visited_fore, visited_back = bastar.searching()
            #找到对应的是extract_path(s_meet), self.CLOSED_fore, self.CLOSED_back
            #path=extract_path(s_meet)=PARENT_fore[s_meet]+PARENT_back[s_meet]
            time4=datetime.datetime.now()
            print('路径规划耗时：%d ms' % ((time4 - time3).seconds * 1000 + (time4 - time3).microseconds / 1000))
            PathPlan_map = bastar.plot.animation_bi_astar(path, visited_fore, visited_back, "Bidirectional-A*")
            #调用 bastar 对象的 plot 属性中的 animation_bi_astar 方法，以执行双向 A* 算法的路径动画绘制。
            #返回的 path 是一个包含路径节点坐标的列表，visited_fore 和 visited_back 则分别记录了前向和后向搜索过程中的访问过的节点。
            #"Bidirectional-A*"：是一个字符串，用作创建动画animation的名称。
            time5=datetime.datetime.now()
            print('绘制路径耗时：%d ms' % ((time5 - time4).seconds * 1000 + (time5 - time4).microseconds / 1000))
            turn_corners=abstract_corners(path)
            guide="直行"
            if 1 in turn_corners:
                corner1 = turn_corners[1]
                print("拐点1的压缩像素坐标：", corner1)
                if corner1[1] > start_node[1]:
                    guide="向右平移"
                elif corner1[1] < start_node[1]:
                    guide="向左平移"
                else:
                    guide="直行"
                print( guide)

            else:
                corner1 = None
                print("没有拐点1",guide)
            



        colors = torch.cat(
            [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        inv_alph_masks = masks * (-mask_alpha) + 1

        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    if args.display_fps:
        # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    
    if num_dets_to_consider == 0:
        # img_numpy = cv2.resize(img_numpy, (PathPlan_map.shape[1], PathPlan_map.shape[0]),
        #                        interpolation=cv2.INTER_NEAREST)  # 把图片大小调整一致
        img_numpy = MapProcessor(img_numpy, 0.3)#图像作整体压缩
        img_numpy = img_numpy.downsample_map

        if 'PathPlan_map'  not in locals(): # 检查变量是否在局部或全局命名空间中
            img_contrast = np.concatenate([img_numpy, img_numpy], axis=1)  # 自己左右拼接
            print("PathPlan_map.不存在")
        else:
            img_contrast = np.concatenate([img_numpy, PathPlan_map], axis=1)  # 进行正常的拼接
            #使用 axis=1 参数表示沿着水平方向拼接
        return img_contrast

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)
    # **************************************************************
    if start_in_obstacle:
        img_numpy = cv2.resize(img_numpy, (PathPlanStop_map.shape[1], PathPlanStop_map.shape[0]),
                               interpolation=cv2.INTER_NEAREST)  # 把图片大小调整一致
        img_contrast = np.concatenate([img_numpy, PathPlanStop_map], axis=1)
    else:
        img_numpy = cv2.resize(img_numpy, (PathPlan_map.shape[1], PathPlan_map.shape[0]),
                               interpolation=cv2.INTER_NEAREST)  # 把图片大小调整一致
        img_contrast = np.concatenate([img_numpy, PathPlan_map], axis=1)
    print("img_contrast_size:", img_contrast.shape)

    return img_contrast


def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


from multiprocessing.pool import ThreadPool
from queue import Queue


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


def evalvideo(net: Yolact, path: str, out_path: str = None):
    # If the path is a digit, parse it as a webcam index
    is_webcam = int(path)

    # If the input image size is constant, this make things faster (hence why we can use it in a video setting).
    cudnn.benchmark = True

    if is_webcam == 0:#如果是单片摄像头
        vid = cv2.VideoCapture(int(path))
        num_frames = float('inf')  # 如果是摄像头，表示要处理无穷多个帧数，即为inf
        target_fps = round(vid.get(cv2.CAP_PROP_FPS))  # 获取视频帧率
        frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度544
        frame_height = round(0.5 * (vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取视频高度480

    elif is_webcam == 1:#如果是realense摄像头
        print("启动realense摄像头")
        num_frames = float('inf')  # 如果是摄像头，表示要处理无穷多个帧数，即为inf
        target_fps=30
        frame_width=640
        frame_height=480

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)     # 配置color流
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)      # 配置depth流

        profile = pipeline.start(config)
        align_to = rs.stream.color  #用于和深度帧对齐的流
        align = rs.align(align_to)  #rs.align 执行深度帧与其他帧的对齐      
    else:
        vid = cv2.VideoCapture(path)
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))  # 将其处理视频的总帧数338
        target_fps = round(vid.get(cv2.CAP_PROP_FPS))  # 获取视频帧率
        frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度544
        frame_height = round(0.5 * (vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取视频高度480
    
    
    def get_next_frame_realsense(pipeline):
        nonlocal profile
        global RGB_frame, depth_frame, depth_intrin, color_intrin, aligned_depth_frame

        frames = []
        aligned_depth_frame=None
        aligned_color_frame=None
        RGB_frame=None
        depth_frame=None
        depth_intrin=None
        color_intrin=None

        for idx in range(args.video_multiframe):
            frameset = pipeline.wait_for_frames()        #获取realsense数据
            aligned_frameset = align.process(frameset)   #获取对齐帧，将深度框与颜色框对齐  

            aligned_depth_frame = aligned_frameset.get_depth_frame()      # 获取对齐帧中的的depth帧 
            aligned_color_frame = aligned_frameset.get_color_frame()      # 获取对齐帧中的的color帧

            #### 获取相机参数 ####
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics     # 获取深度参数（像素坐标系转相机坐标系会用到）
            color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics 
            #####################
        
            if not aligned_color_frame:
                return frames, color_intrin, depth_intrin, RGB_frame,depth_frame, aligned_depth_frame
        
            RGB_frame = np.asanyarray(aligned_color_frame.get_data())
            depth_frame=np.asanyarray(aligned_depth_frame.get_data())
        
            if idx == args.video_multiframe - 2:  # 抽取每组倒数第二帧
                frames.append(RGB_frame)
        return frames, color_intrin, depth_intrin, RGB_frame,depth_frame, aligned_depth_frame

    def get_3d_camera_coordinate(pixel, aligned_depth_frame, depth_intrin):
        dis=""
        x = pixel[0]
        y = pixel[1]
        dis = aligned_depth_frame.get_distance(x, y)        # 获取该像素点对应的深度
        # print ('depth: ',dis)       # 深度单位是m
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, pixel, dis)#相机坐标系下该点的三维坐标
        # print ('camera_coordinate: ',camera_coordinate)
        return dis, camera_coordinate

    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(100)  # 用于跟踪处理每一帧所花费的时间。这里指定了窗口大小为 100。
    fps = 0
    frame_time_target = 1 / target_fps
    running = True
    fps_str = ''
    vid_done = False  # 用于表示视频是否已经处理完成。
    frames_displayed = 0  # 用于跟踪已经显示的帧数。

    if out_path is not None:
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))
 

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            num_extra = 0
            # while imgs.size(0) < args.video_multiframe:
            #     imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
            #     num_extra += 1

            start_time = time.time()
            print('每%f帧取倒数第二帧，规划一次路径' % args.video_multiframe )
            # print('yolact分割了1+%f张图片' % num_extra )
            out = net(imgs)#运行yolact模型，生成预测结果
            end_time = time.time()
            yolact_time = end_time - start_time
            print('yolact模型推理时间为：%f秒' % yolact_time)

            if num_extra > 0:
                out = out[:-num_extra]
            return frames, out

    def prep_frame(inp, fps_str):
        with torch.no_grad():
            frame, preds = inp
            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True, fps_str=fps_str)

    frame_buffer = Queue()
    video_fps = 0

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        if out_path is not None:
            out.release()
        cv2.destroyAllWindows()
        exit()

    # All this timing code to make sure that 
    def play_video():
        try:
            nonlocal frame_buffer, running, video_fps, is_webcam, num_frames, frames_displayed, vid_done
            #nonlocal声明的变量，意味着这些变量并不是局部于这个内部函数的，而是来自于它的外层函数。
            video_frame_times = MovingAverage(100)
            frame_time_stabilizer = frame_time_target
            last_time = None
            stabilizer_step = 0.0005
            progress_bar = ProgressBar(30, num_frames)

            while running:
                frame_time_start = time.time()

                if not frame_buffer.empty():
                    next_time = time.time()
                    if last_time is not None:
                        video_frame_times.add(next_time - last_time)
                        video_fps = 1 / video_frame_times.get_avg()
                    # if out_path is None:
                    #     #cv2.imshow(path, frame_buffer.get())
                    #     cv2.imshow(str(path), frame_buffer.get())
                    if out_path is None:
                        frame00 = frame_buffer.get()
                        # 将显示的图像放大为原来的7倍，你可以根据需要调整放大比例
                        zoomed_frame00 = cv2.resize(frame00, (int(frame00.shape[1] * 4), int(frame00.shape[0] * 4)))
                        cv2.imshow(str(path), zoomed_frame00)
                         #cv2.imshow(str(path), frame_buffer.get())
                    else:
                        out.write(frame_buffer.get())
                    frames_displayed += 1
                    last_time = next_time

                    if out_path is not None:
                        if video_frame_times.get_avg() == 0:
                            fps = 0
                        else:
                            fps = 1 / video_frame_times.get_avg()
                        progress = frames_displayed / num_frames * 100
                        progress_bar.set_val(frames_displayed)

                        print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                              % (repr(progress_bar), frames_displayed, num_frames, progress, fps), end='')

                # This is split because you don't want savevideo to require cv2 display functionality (see #197)
                if out_path is None and cv2.waitKey(1) == 27:
                    # Press Escape to close
                    running = False
                if not (frames_displayed < num_frames):
                    running = False

                if not vid_done:
                    buffer_size = frame_buffer.qsize()
                    if buffer_size < args.video_multiframe:
                        frame_time_stabilizer += stabilizer_step
                    elif buffer_size > args.video_multiframe:
                        frame_time_stabilizer -= stabilizer_step
                        if frame_time_stabilizer < 0:
                            frame_time_stabilizer = 0

                    new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
                else:
                    new_target = frame_time_target

                next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
                target_time = frame_time_start + next_frame_target - 0.001  # Let's just subtract a millisecond to be safe

                if out_path is None or args.emulate_playback:
                    # This gives more accurate timing than if sleeping the whole amount at once
                    while time.time() < target_time:
                        time.sleep(0.001)
                else:
                    # Let's not starve the main thread, now
                    time.sleep(0.001)
        except:
            # See issue #197 for why this is necessary
            import traceback
            traceback.print_exc()

    extract_frame = lambda x, i: (
        x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])
    #x是一个元组，包含两个元素：x[0]是图像，x[1]是字典
    #i是一个整数，表示要处理的图像的索引。
    #lambda函数整句意思是如果没有检测结果那么给出原始图像X【0】，若有检测结果则把图像转移到GPU上然后输出检测结果字典X【1】。
  
    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')
    #first_batch = eval_network(transform_frame(get_next_frame(vid)))  # 是通过执行一系列处理步骤对视频的第一帧进行预处理后得到的结果。这里使用了 transform_frame、 # get_next_frame 和 eval_network 函数来处理第一帧，并将处理后的结果存储在 first_batch 中。
    frames, color_intrin, depth_intrin, RGB_frame,depth_frame, aligned_depth_frame=get_next_frame_realsense(pipeline)
    first_batch = eval_network(transform_frame(frames))  
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]  # 是一个列表，包含了视频帧处理的一系列步骤，按照逆序排列。在每一帧的处理过程中，将按照这个顺序依次执行这些步骤。
    pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)  # 是一个线程池对象，用于并行执行视频帧的处理过程。
    pool.apply_async(play_video)
    active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in
                     range(len(first_batch[0]))]  # 包含了当前活动帧的信息。每个元素都是一个字典，其中 'value' 键存储了帧数据，'idx' 键存储了帧的索引

    print()
    if out_path is None: print('Press Escape to close.')
    try:
        #while vid.isOpened() and running:  # 用于从视频文件中读取帧，并对每一帧进行处理和推断。
        while True and running:  # 用于从视频文件中读取帧，并对每一帧进行处理和推断。
            # Hard limit on frames in buffer so we don't run out of memory >.>
            while frame_buffer.qsize() > 100:
                time.sleep(0.001)

            start_time = time.time()

            # Start loading the next frames from the disk
            if not vid_done:  # 开始加载来自磁盘的下一组帧。如果视频已经结束，则不再加载。
                next_frames = pool.apply_async(get_next_frame_realsense, args=(pipeline,)) #thread-随机性分配
                #next_frames_future = pool.apply_async(get_next_frame_realsense, args=(pipeline,))  # 异步调用
            else:
                next_frames = None

            if not (vid_done and len(active_frames) == 0):
                # For each frame in our active processing queue, dispatch a job
                # for that frame using the current function in the sequence
                for frame in active_frames:
                    _args = [frame['value']]
                    if frame['idx'] == 0:
                        _args.append(fps_str)  # 将FPS信息添加到参数列表中
                    frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)  # 使用线程池异步地执行当前帧的处理函数。thread-11

                # For each frame whose job was the last in the sequence (i.e. for all final outputs)
                for frame in active_frames:
                    if frame['idx'] == 0:
                        frame_buffer.put(frame['value'].get())  # 将处理完的帧放入帧缓冲区

                # Remove the finished frames from the processing queue
                active_frames = [x for x in active_frames if x['idx'] > 0]

                # Finish evaluating every frame in the processing queue and advanced their position in the sequence
                for frame in list(reversed(active_frames)):
                    frame['value'] = frame['value'].get() # .get()用于多线程任务中等待并获取结果。thread-12
                    frame['idx'] -= 1

                    if frame['idx'] == 0:
                        # Split this up into individual threads for prep_frame since it doesn't support batch size
                        active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in
                                          range(1, len(frame['value'][0]))]
                        frame['value'] = extract_frame(frame['value'], 0)

                # Finish loading in the next frames and add them to the processing queue
                if next_frames is not None:
                    #frames = next_frames.get()
                    result = next_frames.get()
                    frames, color_intrin, depth_intrin, RGB_frame,depth_frame, aligned_depth_frame = result
                    
                    if len(frames) == 0:
                        vid_done = True
                    else:
                        active_frames.append({'value': frames, 'idx': len(sequence) - 1})

                # Compute FPS
                frame_times.add(time.time() - start_time)
                fps = args.video_multiframe / frame_times.get_avg()
            else:
                fps = 0

            fps_str = 'Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d' % (
                fps, video_fps, frame_buffer.qsize())
            if not args.display_fps:
                print('\r' + fps_str + '    ', end='')

    except KeyboardInterrupt:
        print('\nStopping...')
    cleanup_and_exit()


def evaluate(net: Yolact, dataset, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    
    evalvideo(net, args.video)
   
    return


def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

    print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps


def print_maps(all_maps):
    # Warning: hacky 
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n: ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()


if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        # if not os.path.exists('results'):
        #     os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.resume and not args.display:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()

        if args.image is None and args.video is None and args.images is None:
            dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                    transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            # prep_coco_cats()
        else:
            dataset = None

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()



        evaluate(net, dataset)
