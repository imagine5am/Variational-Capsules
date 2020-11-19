
import cv2
import h5py
import numpy as np
import os
import random

from PIL import Image, ImageDraw
from scipy.misc import imread
from tqdm import tqdm


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
        draw.polygon(bbox.tolist(), fill=1)
    del draw
    
    # cv2.imwrite('temp2.jpg', im)
    # input()
    return np.asarray(im)

class TrainDataLoader:
    def __init__(self, total_images = 10000):
        self.synth_data_loc = '/mnt/data/Rohit/VideoCapsNet/code/SynthVideo/out'
        synth_data = self.load_synth_data(7000) 
        print(len(synth_data))

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
                data.append((frame, mask. 'synth'))
        
        return data
            
            
if __name__ == "__main__":
    dataloaders = TrainDataLoader()
        
        