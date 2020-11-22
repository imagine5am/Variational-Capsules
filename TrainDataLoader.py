
import cv2
import h5py
import numpy as np
import os
import random
import skvideo.io 
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from scipy.misc import imread
from tqdm import tqdm

out_h, out_w = 480, 480

def save_masked_video(name, video, mask):
    alpha = 0.5
    color = np.zeros((3,)) + [0.0, 0, 1.0]
    masked_vid = np.where(np.tile(mask, [1, 1, 3]) == 1, video * (1 - alpha) + alpha * color, video)
    skvideo.io.vwrite(name+'_segmented.avi', (masked_vid * 255).astype(np.uint8))


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
                     #'word_ann': np.rint(np.array(file_grp.get('word_ann')[()])).astype(np.int32),
                     #'line_ann': np.rint(np.array(file_grp.get('line_ann')[()])).astype(np.int32),
                     'para_ann': np.rint(np.array(file_grp.get('para_ann')[()])).astype(np.int32)
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
        print("Num train samples:", len(train_split))
        return train_split
    elif split == 'test':
        test_split = polygon_ann[-num_test_samples:]
        print("Num test samples:", len(test_split))
        return test_split


def create_mask(shape, pts):
    im = np.zeros(shape, dtype=np.uint8)
    im = Image.fromarray(im, 'L')
    draw = ImageDraw.Draw(im)
    
    for bbox in pts:
        draw.polygon(bbox, fill=1)
    del draw
    
    # cv2.imwrite('temp2.jpg', im)
    # input()
    return np.asarray(im)


def list_vids(dir):
    allfiles = os.listdir(dir)
    files = [ fname for fname in allfiles if fname.endswith('.mp4')]
    return files


def parse_ann(file):
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


class TrainDataLoader:
    def __init__(self, total_images = 10000):
        self.synth_data_loc = '/mnt/data/Rohit/VideoCapsNet/code/SynthVideo/out'
        #synth_data = self.load_synth_data(7000) 
        icdar_data = self.load_icdar_data() # 25 videos 
        #print(len(synth_data))
        print(len(icdar_data))
        print(icdar_data[0][0].shape)
        print(icdar_data[0][1].shape)
        

    def load_synth_data(self, num_images):
        ann_file = os.path.join(self.synth_data_loc, 'Annotations', 'synthvid_ann.hdf5')
        train_files = get_det_annotations(ann_file)
        frames_dir = os.path.join(self.synth_data_loc, 'Frames')
        
        data = []
        
        print('Loading Synthetic training data...')
        for k, v in tqdm(train_files):
            num_frames = v['para_ann'].shape[0]
            image_nums = np.random.choice(num_frames, 2, replace=False)
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
                    
                mask = create_mask((h, w), v['para_ann'][image_num])
                data.append((frame, mask, 'synth'))
        
        return data
    
    
    def load_icdar_data(self, split_type='train'):
        if split_type=='train':
            icdar_loc = '/mnt/data/Rohit/ICDARVideoDataset/text_in_Video/ch3_train/'
        elif split_type=='test':
            icdar_loc = '/mnt/data/Rohit/ICDARVideoDataset/text_in_Video/ch3_test/'
        
        allfiles = list_vids(icdar_loc)
        random.shuffle(allfiles)
        
        data = []
        
        for video_name in tqdm(allfiles):
            ann_file = icdar_loc+video_name[:-4]+'_GT.xml'
            ann = parse_ann(ann_file)

            video_orig = skvideo.io.vread(icdar_loc+video_name)
            num_frames, h, w, _ = video_orig.shape
            print('num_frames:', num_frames)
            chosen_images = np.random.choice(num_frames, num_frames//10, replace=False)
            print('chosen_images:', chosen_images)
            
            for idx in range(chosen_images):
                frame = resize_and_pad((h, w), video_orig[idx])
                print('frame.shape:', frame.shape)
                print('frame.dtype:', frame.dtype)
                
                if idx in ann and ann[idx]:
                    polygons = ann[idx]
                    frame_mask = create_mask((h, w), list(polygons.values()))
                    mask_resized = resize_and_pad((h, w), frame_mask)
                    mask = np.expand_dims(mask_resized, axis=-1)
                else:
                    mask = np.zeros((out_h, out_w, 1), dtype=np.uint8)
                
                data.append((frame, mask, 'icdar'))    
        
        return data
            
if __name__ == "__main__":
    dataloaders = TrainDataLoader()
        
        