import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils as utils
import pickle

from sklearn import preprocessing
from stgcn import STGCN
from utility import generate_dataset, load_metr_la_data, get_normalized_adj, evaluate_model, eval_out_normalized, evaluate_metric, generate_pred_dataset
#Input Shape TaoYuan : Year(12, 4576, 1)  Season(46, 4576, 1)
#Input Shape XinPei : Year(12, 10483, 1)  Season(46, 10483, 1)
torch.cuda.empty_cache()
use_gpu = True
num_timesteps_input = 11  
num_timesteps_output = 1

epochs = 100  
batch_size = 4
learning_rate = 1e-4 

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    print("CUDA")
else:
    args.device = torch.device('cpu')
    print("CPU")


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':

 
    torch.manual_seed(7)

    A, X, means, stds = load_metr_la_data()

    print('X:', X.shape)

    split_line1 = int(X.shape[2] * 0.7)
    print(split_line1)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:]
    print('train_original_data:',train_original_data.shape) #10483, 1, 96
    print('val_original_data:',val_original_data.shape) #10483, 1, 42
    #print(train_original_data)
    #print(val_original_data)
    

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input = generate_pred_dataset(val_original_data,
                                  num_timesteps_input=num_timesteps_input)


    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.to(torch.float32)

    print('training_input type:', training_input.dtype)
    print('training_input shape:', training_input.shape)
    
    A_wave = A_wave.to(device=args.device)
    
    # STGCN(num_nodes, num_features, num_timesteps_input, num_timesteps_output)
    net = STGCN(A_wave.shape[0], 
                training_input.shape[3], 
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #loss_criterion = nn.MSELoss()
    loss_criterion = nn.L1Loss()

    
    training_losses = []
    validation_losses = []
    validation_maes = []
    validation_mses = []
    
    import time
    start = time.time()
    for epoch in range(epochs):
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        print('epoch : ', epoch)

        training_losses.append(loss)

        # Run validation
        with torch.no_grad():
            net.eval()
            
            val_input = val_input.to(device=args.device)
            test_input = test_input.to(device=args.device)
            print('val_input shape', val_input.shape)
            print('test_input:', test_input.shape)
            val_target = val_target.to(device=args.device)

            out = net(A_wave, test_input)
            print('output', out)
            print('output shape', out.shape)
            val_loss = loss_criterion(out, val_target).to(device="cpu")
            validation_losses.append(np.asscalar(val_loss.detach().numpy()))

            out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
            print(out_unnormalized)
            #print('Out_UnNormalized : ', out_unnormalized)
            #print('Out_UnNormalized : ', out_unnormalized.shape)
            target_unnormalized = val_target.detach().cpu().numpy()*stds[0]+means[0]
            
            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            validation_maes.append(mae)
            
            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")
            out_unnormalized, l_sum, grd = eval_out_normalized(out_unnormalized, validation_losses, validation_maes)
            
            np.save('output path/predict_Mat_TY_abs.npy', out_unnormalized)
            
        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAE: {}".format(validation_maes[-1]))
        print("===============================")
        print(" ")

        
        
        torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                    'checkpoints/sampledata_test.pth')
        
        checkpoint_path = "STGCN-Pytorch/checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("STGCN-Pytorch/checkpoints/losses_test" + "_" + str(batch_size) + "_L1loss_"+ str(learning_rate) + ".pk", "wb") as fd:
            pk.dump((training_losses, validation_losses, validation_maes), fd)


    # Move the plot func outside the for loop to prevent redundent windows showing #Lin
    plt.plot(training_losses, label="training loss")
    plt.plot(validation_losses, label="validation loss")
    plt.legend()
    plt.show()