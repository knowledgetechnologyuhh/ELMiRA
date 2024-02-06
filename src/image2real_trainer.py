import torch
from src.image2real import ImageToRealWorldMLP
import os
import numpy as np

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

import datetime
import wandb

def loss(output, ground_truth):
    loss = torch.mean(torch.square(output - ground_truth))
    return loss

def validate(model, batch, epoch_loss_t):
    output = model(batch['input'])
    mse_loss = loss(output, batch['ground_truth'])  # compute loss
    epoch_loss_t.append(mse_loss.item())    # record the batch loss
    return mse_loss # return the loss

def train(model, batch, optimiser, epoch_loss):
    output = model(batch['input'])
    mse_loss = loss(output, batch['ground_truth'])  # compute loss
    mse_loss.backward()  # compute gradients
    optimiser.step()  # update weights
    epoch_loss.append(mse_loss.item())    # record the batch loss
    return mse_loss # return the loss

def main():
    num_of_epochs=100
    log_interval=10
    test=True
    test_interval=1
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
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = MultiStepLR(optimiser, milestones=[10000], gamma=0.5)
    #scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3)
    #  Inspect the model with tensorboard
    model_name = 'xbit_res34_mlpactiondec_three_buttons_120_vars_all_signals_025do'
    date = str(datetime.datetime.now()).split('.')[0]
    #writer = SummaryWriter(log_dir='.././logs/'+model_name+date)  # initialize the writer with folder "./logs"

    # Load the trained model
    #checkpoint = torch.load(save_dir + '/xbit_res34_mlpactiondec_tap_fixed_execonly_pose__500epochs.tar')       # get the checkpoint
    #model.load_state_dict(checkpoint['model_state_dict'])       # load the model state
    #optimiser.load_state_dict(checkpoint['optimiser_state_dict'])   # load the optimiser state

    model.train()  # tell the model that it's training time
    # Load the dataset
    input_coordinates = open('../data/input_coordinates.txt', 'r')
    inputs = input_coordinates.read().splitlines()
    output_coordinates = open('../data/output_coordinates.txt', 'r')
    outputs = output_coordinates.read().splitlines()
    training_data_input = []
    training_data_output = []
    for i, inp in enumerate(inputs):
        row = inp.split('\t')
        row_out = outputs[i].split('\t')
        for j, cell in enumerate(row):
            training_data_input.append(cell.split('-'))
            training_data_output.append(row_out[j].split('-'))
    training_data = {'input':training_data_input, 'ground_truth':training_data_output}
    train_dataloader = DataLoader(training_data, batch_size=5, shuffle=True)
    # Load the dataset
    input_coordinates_val = open('../data/input_coordinates_val.txt', 'r')
    inputs_val = input_coordinates_val.read().splitlines()
    output_coordinates_val = open('../data/output_coordinates_val.txt', 'r')
    outputs_val = output_coordinates_val.read().splitlines()
    val_data_input = []
    val_data_output = []
    for i, inp in enumerate(inputs_val):
        row = inp.split('\t')
        row_out = outputs_val[i].split('\t')
        for j, cell in enumerate(row):
            val_data_input.append(cell.split('-'))
            val_data_output.append(row_out[j].split('-'))
    val_data = {'input':val_data_input, 'ground_truth':val_data_output}
    val_dataloader = DataLoader(val_data, batch_size=5, shuffle=True)
    step = 0
    for epoch in range(num_of_epochs):
        epoch_loss = []
        for input in train_dataloader:
            loss = train(model, input, optimiser, epoch_loss)
            print("step:{} loss:{}".format(step, loss))
        #4. Log the losses
        run.log({"Training Loss": np.mean(epoch_loss)})
        scheduler.step()
        # Validation
        if test and (epoch + 1) % test_interval == 0:
            epoch_loss_t = []
            for input in val_dataloader:
                loss = validate(model, input, epoch_loss_t)
                print("step:{} loss:{}".format(step, loss))
            run.log({"Validation Loss": np.mean(epoch_loss_t)})
        # Save the model parameters at every log interval
        if (epoch+1) % log_interval == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict()},
                       '../image2real_model_'+str(epoch+1)+'epochs.tar')

    wandb.finish()
if __name__ == '__main__':
    main()
