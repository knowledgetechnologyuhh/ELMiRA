import torch.nn as nn

class ImageToRealWorldMLP(nn.Module):
    """The image-to-real-world projection.

    This network is trained to translate image pixel coordinates to real-world coordinates on the table.

    Inputs: 
        x: a 2D vector of image pixel coordinates.

    Args:
        input_size: input size of the first fc layer.
        output_size: output size of the second fc layer.

    Returns:
        y: a 2D vector of real-world coordinates.
    """

    def __init__(self, input_size=2, output_size=2):
        super().__init__()

        self.input_layer = nn.Linear(input_size, int(input_size/2))
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(int(input_size/2), output_size)

        #torch.nn.init.orthogonal_(self.input_layer.weight, np.sqrt(2))    #torch.nn.init.normal_(self.input_layer.weight, mean=0.0, std=1 / np.sqrt(256))
        #torch.nn.init.constant_(self.input_layer.bias, 0.0)
        #torch.nn.init.orthogonal_(self.output_layer.weight)   #torch.nn.init.normal_(self.output_layer.weight, mean=0.0, std=1 / np.sqrt(256))
        #torch.nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x):
        m = self.relu(self.input_layer(x))
        y = self.output_layer(m)
        return y

