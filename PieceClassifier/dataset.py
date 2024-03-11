import json
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import cv2

class SamDataset(Dataset):
    def __init__(self, dataroot='chessReD', classifier='bw', split='train', withbbox=True):
        '''
        dataroot (str): path to the location of dataset. Defaults to './chessReD'.
        classifier (str): data classifier that will use the dataset. 
                            Utilize to determine the correct datalables to use.
                            Available options are:
                                * 'pnp' - piece/not-a-piece classifier **NOT YET IMPLEMENTED**
                                * 'bw' (default) - black/white classifier
                                * 'pt' - Piece type classifier
                                * 'loc' - chessboard location classifier **NOT YET IMPLEMENTED**
        split (str): requested split of the data.
                            Available optins are: 'train', 'val', 'test'
        withbbox (bool): If True - use only images with bounding boxes from the 'chessred2k' section of the dataset

        As of now, the dataset returns a path to the image and the binding box of the piece. 
        It might be usefull (though possibly not required) to have it return the segmented piece itself.
        '''
        self.dataroot = dataroot
        self.classifier = classifier
        self.split = split

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

    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        '''
        Gets a single item from the dataset.
        The item is composed of the path to the image, and the bounding box of the piece
        The label depends on the model the data will be fed to, defined in self.classifier
        '''
        image_path = Path(
            self.dataroot,
            self.images.iloc[self.annotations.iloc[index]['image_id']]['path']
        )
        bbox = self.annotations.iloc[index]['bbox']
        piece_category = self.categories.iloc[self.annotations.iloc[index]['category_id']]['name']
        if self.classifier.lower() in ['bw']:
            label = piece_category.split('-')[0]
        elif self.classifier.lower() in ['p','pt']:
            label = piece_category.split('-')[-1]
        else:
            raise(NotImplementedError(f"Classifier '{self.classifier}' not implemented"))
        return (str(image_path), bbox), label
    