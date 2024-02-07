from image2real import ImageToRealWorldMLP
from object_localiser import OWLv2
import torch

def main():
    #Create the OWLv2 instance
    owlv2 = OWLv2('owlv2')
    #Text and Image input
    text_query = ["Point towards the cube"]  # ["human face", "rocket", "nasa badge", "star-spangled banner"]#
    image_path = '/informatik3/wtm/home/oezdemir/Downloads/nico_examples_higher/picture-2024-01-31T14-51-26.738369.png'  # target_RLBench_three_buttons_120_vars/image_train/230510/target008008/0.png'
    #Get the X, Y pixel space target points
    x, y = owlv2(image_path, text_query, False)
    print("X and Y position of the target object: ", x, y)

    # Use GPU if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    print("The currently selected GPU is number:", torch.cuda.current_device(),
          ", it's a ", torch.cuda.get_device_name(device=None))
    # Create an instance of the image-to-real-world MLP
    image2real_mlp = ImageToRealWorldMLP().to(device)
    # Load the trained model
    checkpoint = torch.load('../model_checkpoints/image2real_model_22000epochs.tar')       # get the checkpoint
    image2real_mlp.load_state_dict(checkpoint['model_state_dict'])       # load the model state
    image2real_mlp.eval()
    # Get the target positions in the real world
    target_positions = image2real_mlp(torch.unsqueeze(torch.FloatTensor([x,y]),0).to(device))
    print("Real-world coordinates of the target object: ", target_positions[0][0].item(), target_positions[0][1].item())

if __name__ == '__main__':
    main()
