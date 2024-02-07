import torch
from src.image2real import ImageToRealWorldMLP, Image2RealDataset
import numpy as np

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

import datetime
import wandb

def loss(output, ground_truth):
    loss = torch.nn.functional.smooth_l1_loss(output, ground_truth, beta=0.01)#torch.mean(torch.square(output - ground_truth))
    return loss

def validate(model, batch, epoch_loss_t):
    with torch.no_grad():
        output = model(batch['input'])
        mse_loss = loss(output, batch['ground_truth'])  # compute loss
        euclidean_distance = torch.linalg.norm(output-batch['ground_truth'], dim=1).mean()
        epoch_loss_t.append(mse_loss.item())    # record the batch loss
    return mse_loss, euclidean_distance # return the loss

def train(model, batch, optimiser, epoch_loss):
    output = model(batch['input'])
    mse_loss = loss(output, batch['ground_truth'])  # compute loss
    mse_loss.backward()  # compute gradients
    optimiser.step()  # update weights
    epoch_loss.append(mse_loss.item())    # record the batch loss
    return mse_loss # return the loss

def main():
    num_of_epochs = 50000
    log_interval = 100
    test = True
    test_interval = 1
    batch_size=5
    # 1. Start a new run for WandB
    run = wandb.init(project="train-image2real")

    # 2. Save model inputs and hyperparameters
    w_and_b_config = run.config

    # Random Initialisation
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    # Use GPU if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    print("The currently selected GPU is number:", torch.cuda.current_device(),
          ", it's a ", torch.cuda.get_device_name(device=None))

    # Create a model instance
    model = ImageToRealWorldMLP().to(device)
    # 3. Log gradients and model parameters
    run.watch(model)
    # Initialise the optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-5)

    scheduler = MultiStepLR(optimiser, milestones=[10000], gamma=0.5)

    model.train()  # tell the model that it's training time
    # Load the dataset
    training_data = Image2RealDataset('../data', 'train')
    val_data = Image2RealDataset('../data', 'val')
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    step = 0
    for epoch in range(num_of_epochs):
        epoch_loss = []
        for input in train_dataloader:
            input['input'] = input['input'].to(device)
            input['ground_truth'] = input['ground_truth'].to(device)
            loss = train(model, input, optimiser, epoch_loss)
            print("step:{} loss:{}".format(step, loss))
            step = step+1
        #4. Log the losses
        run.log({"Training Loss": np.mean(epoch_loss)}, epoch)
        scheduler.step()
        # Validation
        if test and (epoch + 1) % test_interval == 0:
            epoch_loss_t = []
            for input in val_dataloader:
                input['input'] = input['input'].to(device)
                input['ground_truth'] = input['ground_truth'].to(device)
                loss, euclidean_distance = validate(model, input, epoch_loss_t)
                print("validation")
                print("step:{} loss:{}".format(step, loss))
            run.log({"Validation Loss": np.mean(epoch_loss_t)}, epoch)
            run.log({"Validation Mean Euclidean Distance": euclidean_distance}, epoch)
        # Save the model parameters at every log interval
        if (epoch+1) % log_interval == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict()},
                       '../model_checkpoints/image2real_model_'+str(epoch+1)+'epochs.tar')

    wandb.finish()
if __name__ == '__main__':
    main()
