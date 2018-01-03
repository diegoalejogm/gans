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
    
def display_images(images, num_images, epoch, n_batch):
    size_figure_grid = int(math.sqrt(num_images))
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(8, 8))
    
    for k in range(num_images):
        i = k//4
        j = k%4

        v = np.moveaxis(images[k,:].data.cpu().numpy(), 0, -1)
        v_min = v.min(axis=(0, 1), keepdims=True)
        v_max = v.max(axis=(0, 1), keepdims=True)
        v = (v - v_min)/(v_max - v_min)

        ax[i,j].cla()
        ax[i,j].set_axis_off()
        ax[i,j].imshow(v)
    plt.axis('off')
    display.display(plt.gcf())
    
    # Save plot
    out_dir = './data/DC-GAN/images'
    _make_dir(out_dir)
    fig.savefig('{}/epoch_{}_batch_{}.png'.format(out_dir, epoch, n_batch))
    
def save_checkpoint(discriminator, generator, epoch):
    out_dir = './data/DC-GAN/models'
    _make_dir(out_dir)
    torch.save(discriminator.state_dict(), '{}/D_epoch_{}'.format(out_dir, epoch))
    torch.save(generator.state_dict(), '{}/G_epoch_{}'.format(out_dir, epoch))
        
def _make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise