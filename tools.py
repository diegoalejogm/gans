from matplotlib import pyplot as plt
from matplotlib import cm
import math, itertools, os, errno
import numpy as np
import torch
from IPython import display

def print_logs(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):
    print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch, num_epochs, n_batch, num_batches))
    print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error.data[0], g_error.data[0]))
    print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.data.mean(), d_pred_fake.data.mean()))
    
def display_images(images, num_images, epoch, n_batch, data_folder):
    size_figure_grid = int(math.sqrt(num_images))
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(8, 8))
    
    ## Display multiple images in plot
    for k in range(num_images):
        i = k//4
        j = k%4
        
        # Move depth dimension to last
        v = np.moveaxis(images[k,:].data.cpu().numpy(), 0, -1)
        # Remove any dimension that has size=1
        v = np.squeeze(v)
        
        # If image has more than 1 filter
        if len(v.shape) > 2:
            # Rescale image values to range [-1, 1]
            v_min = v.min(axis=(0, 1), keepdims=True)
            v_max = v.max(axis=(0, 1), keepdims=True)
            v = (v - v_min)/(v_max - v_min)
        
        elif len(v.shape) == 1:
            # Reshape vector as 2D matrix
            side = int(math.sqrt(v.shape[0]))
            v = v.reshape(side, side)

        ax[i,j].cla()
        # Remove axis lines
        ax[i,j].set_axis_off()
        
        if len(v.shape) == 3:
            ax[i,j].imshow(v)
        elif len(v.shape) == 2:
            ax[i,j].imshow(v, cmap='Greys')

            
    plt.axis('off')
    display.display(plt.gcf())
    
    # Save plot
    out_dir = '{}/images'.format(data_folder)
    _make_dir(out_dir)
    fig.savefig('{}/epoch_{}_batch_{}.png'.format(out_dir, epoch, n_batch))
    
def save_checkpoint(discriminator, generator, epoch, data_folder):
    out_dir = '{}/models'.format(data_folder)
    _make_dir(out_dir)
    torch.save(discriminator.state_dict(), '{}/D_epoch_{}'.format(out_dir, epoch))
    torch.save(generator.state_dict(), '{}/G_epoch_{}'.format(out_dir, epoch))
    
def _make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise