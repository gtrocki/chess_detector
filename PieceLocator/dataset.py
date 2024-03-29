import cv2
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset
from segment_anything import sam_model_registry, SamPredictor # import relevant parts of the SAM model

#import matplotlib.pyplot as plt
class SamDataset(Dataset):
    def __init__(self, dataroot='chessReD', classifier='bw', crop=True, split='train', imres=[1024,1024], withbbox=True):
        '''
        dataroot (str): path to the location of dataset. Defaults to './chessReD'.
        classifier (str): data classifier that will use the dataset. 
                            Utilize to determine the correct datalables to use.
                            Available options are:
                                * 'pnp' - piece/not-a-piece classifier **NOT YET IMPLEMENTED**
                                * 'bw' (default) - black/white classifier
                                * 'pt' - Piece type classifier
                                * 'loc' - chessboard location classifier
                                * 'pb' - Pawn binary classification
        split (str): requested split of the data.
                            Available optins are: 'train', 'val', 'test'
        withbbox (bool): If True - use only images with bounding boxes from the 'chessred2k' section of the dataset

        As of now, the dataset returns a path to the image and the binding box of the piece. 
        It might be usefull (though possibly not required) to have it return the segmented piece itself.
        '''
        super().__init__()
        self.dataroot = dataroot
        self.classifier = classifier
        self.split = split
        self.imres = imres
        self.crop = crop

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
        
        #Load SAM for both crop=true and crop=false
        sam_checkpoint = Path('C:', '\\', 'Users', 'michalro', 'chess_detector', 'checkpoint_sam', 'sam_vit_h_4b8939.pth') #Check that the checkpoint path suits the computer you are on
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

    def position_to_int(self,position):
        # Converts chess board positions to integers
        file_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
        rank = int(position[1])
        file = file_map[position[0]]
        return (rank - 1) * 8 + (file - 1)
        
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
        
        

        if not self.crop:
            original_resolution = image.shape[:2]
            image = cv2.resize(image, self.imres)
        else:
            original_resolution = self.imres #this makes sure that no reshape is performed when you use the crops only. not sure that this is the best way...

        # Transform the bbox to match SAMs requirements
        bbox = self.annotations.iloc[index]['bbox']
        bbox = torch.tensor(self._bbox_transform(bbox, original_resolution)).to(self.device)
        
        if self.crop:
            x = bbox.int()
            y = cv2.resize(image[x[1]:x[3],x[0]:x[2]], [256,256]) #get the crop of the image and resize to fit into resnet
            data = {'image':image,'bb':x,'out':y}

            
        else:
            input_box = self.predictor.transform.apply_boxes_torch(bbox, image.shape[:2])
    
            # Load the image to SAM and predict segment
            self.predictor.set_image(image)
            masks, scores, logits = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=input_box,
                multimask_output=False,
            )

            """
            binary_tensor = masks.to(torch.uint8)
            x = bbox.int()
            h, w = binary_tensor.shape[-2:]
            binary_tensor = binary_tensor.reshape(h, w, 1)
            MaskedImage = binary_tensor*image"""
            #MaskedImage = image[masks]
            #x = bbox.int()
            x = bbox.int()
            imageTensor = torch.tensor(image).to(self.device)
            h, w = masks.shape[-2:]
            masks = masks.reshape(h, w, 1)
            
            

            if self.classifier != 'loc':
                MaskedImage = imageTensor*masks
                invertedMask = ~masks
                invertedMask = torch.cat((invertedMask,invertedMask,invertedMask),dim=2)
                newBackgroundvalue = 128
                MaskedImage[invertedMask] = newBackgroundvalue
                Cropped_masks = MaskedImage[x[1]:x[3],x[0]:x[2]]
                #resize the cropped image
                tensor_np = Cropped_masks.numpy()
                resized_image = cv2.resize(tensor_np, (256, 256)) 
                data = torch.tensor(resized_image)
            else:
                channel_index = 2 # Choose the blue channel.
                channel_index2 = 1
                invertedMask = ~masks

                # Split the image tensor into a list of individual channels.
                channels = list(torch.chunk(imageTensor, dim=2, chunks=imageTensor.size(1)))

                # Apply the mask to the blue channel.
                newBackgroundvalue = 128
                masked_channel = channels[2]
                masked_channel2 = channels[1]
                masked_channel[invertedMask] = newBackgroundvalue
                masked_channel2[invertedMask] = newBackgroundvalue

                # Replace the masked channel in the list of channels.
                channels[channel_index] = masked_channel
                channels[channel_index2] = masked_channel2

                # Concatenate the channels back into a single tensor.
                masked_image_tensor = torch.cat(channels, dim=2)
                
                #resize the image.
                tensor_np = masked_image_tensor.cpu().numpy()
                resized_image = cv2.resize(tensor_np, (224, 224)) 
                data = torch.tensor(resized_image).to(self.device)
        
        
            
            


    
        # get piece category
        piece_category = self.categories.iloc[self.annotations.iloc[index]['category_id']]['name']
        if self.classifier.lower() in ['bw']:
            encoding = {'black':0,'white':1}
            label = encoding[piece_category.split('-')[0]] # 1st parth of the category is the color
        elif self.classifier.lower() in ['p','pt']:
            encoding = {'pawn':0, 'rook':1, 'knight':2, 'bishop':3, 'queen':4, 'king':5, 'empty':6}
            label = encoding[piece_category.split('-')[-1]] # 2nd part of the category is the piece type
        elif self.classifier.lower() in ['pb']:
            encoding = {'pawn':0, 'rook':1, 'knight':1, 'bishop':1, 'queen':1, 'king':1, 'empty':1}
            label = encoding[piece_category.split('-')[-1]] # 2nd part of the category is the piece type    
        elif self.classifier.lower() in ['loc','locator']:
            label = self.annotations.iloc[index]['chessboard_position'] # get the piece location
            label = self.position_to_int(label)
        else:
            raise(NotImplementedError(f"Classifier '{self.classifier}' not implemented"))

        
        return data, label

