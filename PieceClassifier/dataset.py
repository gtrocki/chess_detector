import cv2
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from segment_anything import sam_model_registry, SamPredictor # import relevant parts of the SAM model

class SamDataset(Dataset):
    def __init__(self, dataroot='chessReD', classifier='bw', split='train', imres=[1024,1024], withbbox=True):
        '''
        dataroot (str): path to the location of dataset. Defaults to './chessReD'.
        classifier (str): data classifier that will use the dataset. 
                            Utilize to determine the correct datalables to use.
                            Available options are:
                                * 'pnp' - piece/not-a-piece classifier **NOT YET IMPLEMENTED**
                                * 'bw' (default) - black/white classifier
                                * 'pt' - Piece type classifier
                                * 'loc' - chessboard location classifier
        split (str): requested split of the data.
                            Available optins are: 'train', 'val', 'test'
        withbbox (bool): If True - use only images with bounding boxes from the 'chessred2k' section of the dataset

        As of now, the dataset returns a path to the image and the binding box of the piece. 
        It might be usefull (though possibly not required) to have it return the segmented piece itself.
        '''
        self.dataroot = dataroot
        self.classifier = classifier
        self.split = split
        self.imres = imres

        data_path=Path(dataroot, 'annotations.json')
        if not data_path.is_file():
            raise(FileNotFoundError(f"File '{data_path}' doesn't exist."))
        with open(data_path, 'r') as f:
            annotations_file = json.load(f)
            
        # Load data tables
        self.annotations = pd.DataFrame(
            annotations_file['annotations']['pieces']
        )
        self.images = pd.DataFrame(
            annotations_file['images']
        )
        self.categories = pd.DataFrame(
            annotations_file['categories']
        )

        # Get the indexes of the images corresponding to the split
        if withbbox:
            self.split_img_ids = annotations_file['splits']['chessred2k'][split]['image_ids']
        else:
            raise(NotImplementedError(f"Dataset implemented only for images with bounding boxes"))
            # self.split_img_ids = annotations_file['splits'][split]['image_ids']

        # Filter data tables to keep only relevant parts
        self.annotations = self.annotations[self.annotations["image_id"].isin(self.split_img_ids)]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # set the device according to availability
        
        sam_checkpoint = Path('/work/amirkl/DL4CV/segment-anything', 'sam_vit_h_4b8939.pth') #Check that the checkpoint path suits the computer you are on
        model_type = 'vit_h'
        
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # create SAM model
        self.sam.to(self.device) # move SAM to device
        
        self.predictor = SamPredictor(self.sam) # create SAM predictor

    def _bbox_transform(self, bbox, original_resolution):
        # utility function for transforming box coordinates in (x,y,w,h) to (x1,y1,x2,y2)
        x_factor = self.imres[0] / original_resolution[0]
        y_factor = self.imres[1] / original_resolution[1]
        bbox[0] = bbox[0] * x_factor
        bbox[1] = bbox[1] * y_factor
        bbox[2] = bbox[2] * x_factor
        bbox[3] = bbox[3] * y_factor
        return [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        '''
        Gets a single item from the dataset.
        The item is composed of the path to the image, and the bounding box of the piece
        The label depends on the model the data will be fed to, defined in self.classifier
        '''
        # get image path and load image
        image_path = str(Path(self.dataroot, self.images.iloc[self.annotations.iloc[index]['image_id']]['path']))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_resolution = image.shape[:2]
        image = cv2.resize(image, self.imres)

        # Transform the bbox to match SAMs requirements
        bbox = self.annotations.iloc[index]['bbox']
        bbox = torch.tensor(self._bbox_transform(bbox, original_resolution)).to(self.device)
        input_box = self.predictor.transform.apply_boxes_torch(bbox, image.shape[:2])

        # Load the image to SAM and predict segment
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_box,
            multimask_output=False,
        )

        # get piece category
        piece_category = self.categories.iloc[self.annotations.iloc[index]['category_id']]['name']
        if self.classifier.lower() in ['bw']:
            label = piece_category.split('-')[0] # 1st parth of the category is the color
            data = {'masks':masks, 'image_path':image_path, 'bbox':bbox}
        elif self.classifier.lower() in ['p','pt']:
            label = piece_category.split('-')[-1] # 2nd part of the category is the piece type
            data = {'masks':masks, 'image_path':image_path, 'bbox':bbox}
        elif self.classifier.lower() in ['loc','locator']:
            label = self.annotations.iloc[index]['chessboard_position']
            data = {'masks':masks, 'image_path':image_path, 'bbox':bbox}
        else:
            raise(NotImplementedError(f"Classifier '{self.classifier}' not implemented"))

        
        return data, label

