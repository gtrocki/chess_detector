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
      nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      
      )   
    
    def forward(self, x):
        """ Apply the forward pass of the network.
        
        Args:
        x (torch.Tensor): masked image input.
        Has shape `(batch_size, num_channels, height, width)`.

        Returns:
        class : piece or no piece.
        
        """    
        y = self.model(x)
        return y
    

