
import cv2
import h5py
import json
import numpy as np
import os
import random
import re
import skvideo.io 
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from PIL.ImageShow import show
from scipy.misc import imread, imsave, imshow
from torch.utils.data import Dataset
from tqdm import tqdm

out_h, out_w = 480, 480
debug_dir = './debug'
DEBUG = True

if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)

def order_points(pts):
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "int32")

    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect.flatten().tolist()

def resize_and_pad(shape, im):
    in_h, in_w = shape[0], shape[1]
    if out_w / out_h > in_w / in_h:
        h, w = out_h, in_w * out_h // in_h
    elif out_w / out_h < in_w / in_h:
        h, w =  in_h * out_w // in_w, out_w

    im = cv2.resize(im, (w, h))
    delta_w = out_w - w
    delta_h = out_h - h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return im


def get_det_annotations(ann_file, split='train'):
    """
    Loads in all annotations for training and testing splits. This is assuming the data has been loaded in with resized frames (half in each dimension).
    :return: returns two lists (one for training and one for testing) with the file names and annotations. The lists
    contain tuples with the following content (file name, annotations), where annotations is a list of tuples with the
    form (start frame, end frame, label, bounding boxes).
    """
    polygon_ann = []
    with h5py.File(ann_file, 'r') as hf:
        for label in hf.keys():
            label_grp = hf.get(label)
            for file in label_grp.keys():
                file_grp = label_grp.get(file)
                k = label + '/' + file
                v = {'label': int(label),
                     #'char_ann': np.rint(np.array(file_grp.get('char_ann')[()])).astype(np.int32),
                     'word_ann': np.rint(np.array(file_grp.get('word_ann')[()])).astype(np.int32),
                     #'line_ann': np.rint(np.array(file_grp.get('line_ann')[()])).astype(np.int32),
                     #'para_ann': np.rint(np.array(file_grp.get('para_ann')[()])).astype(np.int32)
                    }
                polygon_ann.append((k, v))
    random.seed(7)
    random.shuffle(polygon_ann)
    num_samples = len(polygon_ann)
    num_train_samples = int(0.8 * num_samples)
    num_test_samples = num_samples - num_train_samples
    
    if split == 'train':
        train_split = polygon_ann[:num_train_samples]
        random.seed()
        random.shuffle(train_split)
        print(f'Num of synthetic train videos: {len(train_split)}')
        return train_split
    elif split == 'test':
        test_split = polygon_ann[-num_test_samples:]
        print(f'Num of synthetic test videos: {len(test_split)}')
        return test_split


def create_mask(shape, pts, is_rectangle=False):
    mask = np.zeros(shape, dtype=np.uint8)
    mask = Image.fromarray(mask, 'L')
    draw = ImageDraw.Draw(mask)
    
    fill_val = 1
    
    for pt in pts:
        
        if is_rectangle:
            print(pt)
            draw.rectangle(pt, fill=fill_val)
            
        else:
            if isinstance(pt, np.ndarray):
                draw.polygon(pt.tolist(), fill=fill_val)
            else:
                draw.polygon(pt, fill=fill_val)
    del draw
    # show(mask)
    mask = np.asarray(mask).copy()
    return mask


def list_vids(dir):
    allfiles = os.listdir(dir)
    files = [ fname for fname in allfiles if fname.endswith('.mp4')]
    return files


def parse_icdar_ann(file):
    '''
    Returns a dict which is something like:
    {frame_num:{object_id: polygon_pts_list, ...}, ...}
    '''
    tree = ET.parse(file)
    root = tree.getroot()
    ann = {}
    for frame in root.findall('./frame'):
        frame_num = int(frame.attrib['ID']) - 1
        objects = {}
        for object in frame.findall('./object'):
            id = int(object.attrib['ID'])
            pts = []
            for pt in object.findall('./Point'):
                pts.append((int(pt.attrib['x']), int(pt.attrib['y'])))
            objects[id] = order_points(np.array(pts))
        ann[frame_num] = objects
    return ann


def process_roadtext_ann(ann_file):  
    with open(ann_file) as f:
        data = json.load(f)
    
    data_dict = {}  # {<video_num1>:{0: [..], 1: [..], ..}}
    
    for item in data:
        if item['labels']:
            video_num = int(item['videoName'])
            x1 = re.search(r'\d+\.jpg', item['name']) # "name": "dataset_imgs/99-0000280.jpg",
            page_num = int(item['name'][x1.start():x1.end()-4]) - 1

            if video_num not in data_dict:
                data_dict[video_num] = {}

            data_dict[video_num][page_num] = {}
            
            for label in item['labels']:
                box = label['box2d']
                pts = [round(box['x1']), round(box['y1']), round(box['x2']), round(box['y2'])]
                data_dict[video_num][page_num][label['id']] = pts
                
    return data_dict


class CustomDataset (Dataset):
    def __init__(self, split_type='train'):
        np.random.seed(7)
        
        icdar_data = self.load_icdar_data(split_type)
        print(f'len(icdar_data): {len(icdar_data)}')
        
        roadtext_data = self.load_roadtext_data(split_type)
        print(f'len(roadtext_data): {len(roadtext_data)}')
        
        if split_type == 'train':
            synth_data = self.load_synth_data()
            print(f'len(synth_data): {len(synth_data)}')
            
            if DEBUG:
                self.debug_data(synth_data=synth_data, icdar_data=icdar_data, roadtext_data=roadtext_data)

            self.data = synth_data + icdar_data + roadtext_data
            
        else: 
            if DEBUG:
                # self.debug_data(roadtext_data=roadtext_data)
                self.debug_data(icdar_data=icdar_data, roadtext_data=roadtext_data)
                
            # self.data = roadtext_data
            self.data = icdar_data + roadtext_data
        
        random.shuffle(self.data)
       
     
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        image, mask, dataset = self.data[idx] 
        return image / 255., mask
    

    def load_synth_data(self, split_type='train'):
        synth_data_loc = '/mnt/data/Rohit/VideoCapsNet/code/SynthVideo/out'
        
        ann_file = os.path.join(synth_data_loc, 'Annotations', 'synthvid_ann.hdf5')
        train_files = get_det_annotations(ann_file, split_type)
        frames_dir = os.path.join(synth_data_loc, 'Frames')
        
        data = []
        
        print('Loading Synthetic training data...')
        
        for k, v in tqdm(train_files):
            num_frames = v['word_ann'].shape[0]
            image_nums = np.random.choice(num_frames, 1, replace=False)
            video_dir = os.path.join(frames_dir, k)
            
            im0 = imread(os.path.join(video_dir, 'frame_%d.jpg' % 0))
            h, w, ch = im0.shape
            
            for image_num in image_nums:
                frame = imread(os.path.join(video_dir, 'frame_%d.jpg' % image_num))
                
                if (h, w, ch) != frame.shape:
                    print('BAD FRAMES FOUND')
                    print('Video:', video_dir)
                    print('Frame:', image_num)
                    print('*' * 20)
                    frame = cv2.resize(frame, (w, h))
                    
                mask = create_mask((h, w), v['word_ann'][image_num])
                
                frame_resized = resize_and_pad((h, w), frame)
                mask_resized = resize_and_pad((h, w), mask)
                mask = np.expand_dims(mask_resized, axis=-1)
                
                data.append((frame_resized, mask, 'synth'))
        
        return data
    
    
    def load_icdar_data(self, split_type='train'):
        if split_type=='train': # 25 videos 
            
            print('Loading ICDAR training data...')
            icdar_loc = '/mnt/data/Rohit/ICDARVideoDataset/text_in_Video/ch3_train/'
            selection_ratio = 0.5
        
        elif split_type=='test':
            
            print('Loading ICDAR test data...')
            icdar_loc = '/mnt/data/Rohit/ICDARVideoDataset/text_in_Video/ch3_test/'
            selection_ratio = 1
            
        allfiles = list_vids(icdar_loc)
        random.shuffle(allfiles)
        
        data = []
        
        for video_name in tqdm(allfiles):
            ann_file = icdar_loc+video_name[:-4]+'_GT.xml'
            ann = parse_icdar_ann(ann_file)

            video_orig = skvideo.io.vread(os.path.join(icdar_loc, video_name))
            num_frames, h, w, _ = video_orig.shape
            chosen_frames = np.random.choice(num_frames, int(selection_ratio * num_frames), replace=False)
            
            for idx in chosen_frames:
                frame = resize_and_pad((h, w), video_orig[idx])
                # imshow(frame)
                
                if idx in ann and ann[idx]:
                    polygons = list(ann[idx].values())
                    frame_mask = create_mask((h, w), polygons)
                    mask_resized = resize_and_pad((h, w), frame_mask)
                    mask = np.expand_dims(mask_resized, axis=-1)
                    # imshow(frame_mask)
                else:
                    # print(f'ann not in {idx} - {video_name}')
                    mask = np.zeros((out_h, out_w, 1), dtype=np.uint8)
                
                data.append((frame, mask, 'icdar'))    
        
        return data
    
    
    def load_roadtext_data(self, split_type='train'):
        
        print(f'Loading Roadtext {split_type} data...')
        ann_loc = '/mnt/data/Rohit/VideoCapsNet/data/RoadText-1K/Ground_truths/Localisation'
        video_dir = os.path.join('/mnt/data/Rohit/VideoCapsNet/data/RoadText-1K', split_type)
        
        
        if split_type == 'train':
            selection_ratio = 0.5
        else:
            selection_ratio = 1
        
        retVal = []
        
        for dir in os.listdir(video_dir)[:1]:
            print(f'Loading videos in {dir}...')
            ann_file = os.path.join(ann_loc, dir+'_videos_results.json')
            ann = process_roadtext_ann(ann_file)

            curr_dir = os.path.join(video_dir, dir)
                        
            for video_name in tqdm(os.listdir(curr_dir)[:20]):
                
                vid_num = int(video_name[:-4])
                vid_file = os.path.join(curr_dir, video_name)
                video_orig = skvideo.io.vread(vid_file)
                
                num_frames, h, w, _ = video_orig.shape
                chosen_frames = np.random.choice(num_frames, int(selection_ratio * num_frames), replace=False)
                
                for idx in chosen_frames:
                    frame = resize_and_pad((h, w), video_orig[idx])
                    # imshow(frame)
                    
                    if idx in ann and ann[idx]:
                        
                        frame_mask = create_mask((h, w), ann[vid_num][idx], is_rectangle=True)
                        mask_resized = resize_and_pad((h, w), frame_mask)
                        mask = np.expand_dims(mask_resized, axis=-1)
                        # imshow(frame_mask)
                    else:
                        # print(f'ann not in {idx} - {video_name}')
                        mask = np.zeros((out_h, out_w, 1), dtype=np.uint8)

                    retVal.append((frame, mask, 'roadtext'))   
                    
        return retVal
    
    
    '''
    def load_CAT_data(self, split_type='train'):
        if split_type == 'train':
            cat_ann_loc = "/mnt/data/Rohit/ACMData/1a_CATVideosTrain/1ShubhamTrainclean"
        elif split_type == 'val':
            cat_ann_loc = "/mnt/data/Rohit/ACMData/1a_CATVideosTrain/2ShubhamValclean"
        elif split_type == 'test':
            cat_ann_loc = "/mnt/data/Rohit/ACMData/1a_CATVideosTrain/3ShubhamTestclean"
    '''
    
    
    def debug_data(self, synth_data=None, icdar_data=None, roadtext_data=None):
        sample_size = 200
        apply_mask = False
        
        dataset_dict = {'synth': synth_data,
                   'icdar': icdar_data, 
                   'roadtext': roadtext_data
                   }
            
        
        for dataset_name, dataset in dataset_dict.items():
            
            if dataset:
                samples = np.random.choice(len(dataset), sample_size, replace=False)
                
                base_loc = os.path.join(debug_dir, dataset_name)
                if not os.path.exists(base_loc):
                    os.makedirs(base_loc)
                
                for idx in samples:
                    frame, mask, _ = dataset[idx]

                    if apply_mask:
                        save_loc = os.path.join(base_loc, str(idx)+'_applied_mask.jpg')
                        imsave(save_loc, frame * mask)
                    else:
                        frame_save_loc = os.path.join(base_loc, str(idx)+'_frame.jpg')
                        mask_save_loc = os.path.join(base_loc, str(idx)+'_mask.jpg')
                        imsave(frame_save_loc, frame)
                        mask = np.squeeze(mask, axis=-1) * 255
                        imsave(mask_save_loc, mask)
                    

if __name__ == "__main__":
    DEBUG = True
    
    dataset = CustomDataset(split_type='test')
    
        