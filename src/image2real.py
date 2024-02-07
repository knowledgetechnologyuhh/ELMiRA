import torch.nn as nn
from torch.utils.data import Dataset
import torch

class Image2RealDataset(Dataset):
    def __init__(self, data_path, split='train'):
        data_path = data_path
        if split == 'train':
            input_coordinates = open(data_path+'/input_coordinates.txt', 'r')
            output_coordinates = open(data_path+'/output_coordinates.txt', 'r')
        else:
            input_coordinates = open(data_path+'/input_coordinates_val.txt', 'r')
            output_coordinates = open(data_path+'/output_coordinates_val.txt', 'r')
        inputs = input_coordinates.read().splitlines()
        outputs = output_coordinates.read().splitlines()
        self.input = []
        self.output = []
        for i, inp in enumerate(inputs):
            row = inp.split('\t')
            row_out = outputs[i].split('\t')
            for j, cell in enumerate(row):
                self.input.append(torch.tensor(list(map(float, cell.split('-'))), dtype=torch.float32)/100.0)
                self.output.append(torch.tensor(list(map(float, row_out[j].split('_'))), dtype=torch.float32))
    def __len__(self):
        return len(self.training_data_input)

    def __getitem__(self, index):
        items = {}
        items["input"] = self.input[index]
        items["ground_truth"] = self.output[index]
        return items


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

        self.input_layer = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.middle_layer = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, output_size)

        #torch.nn.init.orthogonal_(self.input_layer.weight, np.sqrt(2))    #torch.nn.init.normal_(self.input_layer.weight, mean=0.0, std=1 / np.sqrt(256))
        #torch.nn.init.constant_(self.input_layer.bias, 0.0)
        #torch.nn.init.orthogonal_(self.output_layer.weight)   #torch.nn.init.normal_(self.output_layer.weight, mean=0.0, std=1 / np.sqrt(256))
        #torch.nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x):
        m = self.relu(self.middle_layer(self.relu(self.input_layer(x))))
        y = self.output_layer(m)
        return y

