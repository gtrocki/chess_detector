from torch import nn
import utils
from functools import partial

############################
# Piece no piece classifier
############################

class PnoP(nn.Module):
  """A piece no piece """

  def __init__(self, kernel_size=3, stride=1, padding=0):
    """
    Args:
      kernel_size (int): size of kernels to use in convolutions.
    """
    
    super(PnoP, self).__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    self.model = nn.Sequential(
      nn.Conv2d(3, 4, kernel_size=7, stride=2, padding=3),
      nn.ReLU(inplace=True),
      nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True), #up to this step the shape should be (batch,16,64,64)
      nn.MaxPool2d(kernel_size=8, stride=8,padding=1), #shape (batch,16,8,8)
      
      )
    self.linear = nn.Linear(16*8*8,2,bias=True)
    
    def forward(self, x):
        """ Apply the forward pass of the network.
        
        Args:
        x (torch.Tensor): masked image input.
        Has shape `(batch_size, num_channels, height, width)`.

        Returns:
        class : piece or no piece.
        
        """  
        batch_size = x.shape[0]  
        Feature_maps = self.model(x)
        Feature_maps_vector = Feature_maps.view(batch_size, -1)
        y = self.linear(Feature_maps_vector)
        y = nn.Softmax(dim=1)
        return y
    

