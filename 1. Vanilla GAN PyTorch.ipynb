{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/diego/.pyenv/versions/miniconda-latest/lib/python2.7/site-packages/subprocess32.py:472: RuntimeWarning: The _posixsubprocess module is not being used. Child process reliability may suffer if your program uses threads.\n",
      "  \"program uses threads.\", RuntimeWarning)\n"
     ]
    }
   ],
   "source": [
    "%load_ext autoreload\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "%autoreload 2\n",
    "\n",
    "from IPython import display\n",
    "\n",
    "from utils import Logger\n",
    "\n",
    "import torch\n",
    "from torch import nn, optim\n",
    "from torch.autograd.variable import Variable\n",
    "from torchvision import transforms, datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "DATA_FOLDER = './torch_data/VGAN/MNIST'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mnist_data():\n",
    "    compose = transforms.Compose(\n",
    "        [transforms.ToTensor(),\n",
    "         transforms.Normalize((.5, .5, .5), (.5, .5, .5))\n",
    "        ])\n",
    "    out_dir = '{}/dataset'.format(DATA_FOLDER)\n",
    "    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\n",
    "data = mnist_data()\n",
    "# Create loader with data, so that we can iterate over it\n",
    "data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)\n",
    "# Num batches\n",
    "num_batches = len(data_loader)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Networks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "class DiscriminatorNet(torch.nn.Module):\n",
    "    \"\"\"\n",
    "    A three hidden-layer discriminative neural network\n",
    "    \"\"\"\n",
    "    def __init__(self):\n",
    "        super(DiscriminatorNet, self).__init__()\n",
    "        n_features = 784\n",
    "        n_out = 1\n",
    "        \n",
    "        self.hidden0 = nn.Sequential( \n",
    "            nn.Linear(n_features, 1024),\n",
    "            nn.LeakyReLU(0.2),\n",
    "            nn.Dropout(0.3)\n",
    "        )\n",
    "        self.hidden1 = nn.Sequential(\n",
    "            nn.Linear(1024, 512),\n",
    "            nn.LeakyReLU(0.2),\n",
    "            nn.Dropout(0.3)\n",
    "        )\n",
    "        self.hidden2 = nn.Sequential(\n",
    "            nn.Linear(512, 256),\n",
    "            nn.LeakyReLU(0.2),\n",
    "            nn.Dropout(0.3)\n",
    "        )\n",
    "        self.out = nn.Sequential(\n",
    "            torch.nn.Linear(256, n_out),\n",
    "            torch.nn.Sigmoid()\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.hidden0(x)\n",
    "        x = self.hidden1(x)\n",
    "        x = self.hidden2(x)\n",
    "        x = self.out(x)\n",
    "        return x\n",
    "    \n",
    "def images_to_vectors(images):\n",
    "    return images.view(images.size(0), 784)\n",
    "\n",
    "def vectors_to_images(vectors):\n",
    "    return vectors.view(vectors.size(0), 1, 28, 28)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "class GeneratorNet(torch.nn.Module):\n",
    "    \"\"\"\n",
    "    A three hidden-layer generative neural network\n",
    "    \"\"\"\n",
    "    def __init__(self):\n",
    "        super(GeneratorNet, self).__init__()\n",
    "        n_features = 100\n",
    "        n_out = 784\n",
    "        \n",
    "        self.hidden0 = nn.Sequential(\n",
    "            nn.Linear(n_features, 256),\n",
    "            nn.LeakyReLU(0.2)\n",
    "        )\n",
    "        self.hidden1 = nn.Sequential(            \n",
    "            nn.Linear(256, 512),\n",
    "            nn.LeakyReLU(0.2)\n",
    "        )\n",
    "        self.hidden2 = nn.Sequential(\n",
    "            nn.Linear(512, 1024),\n",
    "            nn.LeakyReLU(0.2)\n",
    "        )\n",
    "        \n",
    "        self.out = nn.Sequential(\n",
    "            nn.Linear(1024, n_out),\n",
    "            nn.Tanh()\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.hidden0(x)\n",
    "        x = self.hidden1(x)\n",
    "        x = self.hidden2(x)\n",
    "        x = self.out(x)\n",
    "        return x\n",
    "    \n",
    "# Noise\n",
    "def noise(size):\n",
    "    n = Variable(torch.randn(size, 100))\n",
    "    if torch.cuda.is_available(): return n.cuda() \n",
    "    return n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "discriminator = DiscriminatorNet()\n",
    "generator = GeneratorNet()\n",
    "if torch.cuda.is_available():\n",
    "    discriminator.cuda()\n",
    "    generator.cuda()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Optimization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Optimizers\n",
    "d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)\n",
    "g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)\n",
    "\n",
    "# Loss function\n",
    "loss = nn.BCELoss()\n",
    "\n",
    "# Number of steps to apply to the discriminator\n",
    "d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1\n",
    "# Number of epochs\n",
    "num_epochs = 200"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def real_data_target(size):\n",
    "    '''\n",
    "    Tensor containing ones, with shape = size\n",
    "    '''\n",
    "    data = Variable(torch.ones(size, 1))\n",
    "    if torch.cuda.is_available(): return data.cuda()\n",
    "    return data\n",
    "\n",
    "def fake_data_target(size):\n",
    "    '''\n",
    "    Tensor containing zeros, with shape = size\n",
    "    '''\n",
    "    data = Variable(torch.zeros(size, 1))\n",
    "    if torch.cuda.is_available(): return data.cuda()\n",
    "    return data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_discriminator(optimizer, real_data, fake_data):\n",
    "    # Reset gradients\n",
    "    optimizer.zero_grad()\n",
    "    \n",
    "    # 1.1 Train on Real Data\n",
    "    prediction_real = discriminator(real_data)\n",
    "    # Calculate error and backpropagate\n",
    "    error_real = loss(prediction_real, real_data_target(real_data.size(0)))\n",
    "    error_real.backward()\n",
    "\n",
    "    # 1.2 Train on Fake Data\n",
    "    prediction_fake = discriminator(fake_data)\n",
    "    # Calculate error and backpropagate\n",
    "    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))\n",
    "    error_fake.backward()\n",
    "    \n",
    "    # 1.3 Update weights with gradients\n",
    "    optimizer.step()\n",
    "    \n",
    "    # Return error\n",
    "    return error_real + error_fake, prediction_real, prediction_fake\n",
    "\n",
    "def train_generator(optimizer, fake_data):\n",
    "    # 2. Train Generator\n",
    "    # Reset gradients\n",
    "    optimizer.zero_grad()\n",
    "    # Sample noise and generate fake data\n",
    "    prediction = discriminator(fake_data)\n",
    "    # Calculate error and backpropagate\n",
    "    error = loss(prediction, real_data_target(prediction.size(0)))\n",
    "    error.backward()\n",
    "    # Update weights with gradients\n",
    "    optimizer.step()\n",
    "    # Return error\n",
    "    return error"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Generate Samples for Testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_test_samples = 16\n",
    "test_noise = noise(num_test_samples)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Start training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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9zi/GS913+EcoC9hHdTZnnLKKal+1llnlTI1cxw7XUuMl3X7DHWqjImqC9LYGVH19tX+jshevy+++GJq4z6vfcq94Nprr0111WwxZnDs7rvvvlKmnzPflfch1EEdnO51jJ+802DmzJmlvP7666c2asN036fnLjVzuu6oe+O85Dq78cYbS/mggw5KbYyX2sf0IaQPZLdu3UqZ8ZGaV+2nO++8M7Xx/1IPTq1tHTpn6PdJv2QdH/XfjajOJz1fcs1xX1SdKrWyqjWNqHp4qk8nvXv5/A888EApU3vKNXjeeeeVMvdQ3v2g+nvqMXmngd5BEpE1o18mXlJXPnLkyFRXv1U+0913353qOq5cczyvqZaTa453nzBG6hmMcUz7OyL7PdMzmOtMtf1cy1dddVWqa7zkmmO85LlWtfDcr75MvPyi+C+ixhhjjDHGGGOain+IGmOMMcYYY4xpKktMai5TLnglsF4nz9QCpunwymZtZyqHpvBE5NRWXudM2xhNA2Aq3yuvvJLq+qdufg5tSjQNcI899khtalkSUU3105QqWiBoenBEvq59q622Sm1Md2PaiMJ0N03HUjuciOpV4trfETmdmCkBTC+49957S5npG7zGXucP+5ApDWqpwev6mSqn144zXY/WOkzx0bFjGhSfqUePHqXMtG99t4jch3XXokfklGBaEPFzOSe0L5i6wmvrdY4w3ZZ2CZpCyPnN8VDrAqZj06KCY6epl0x/O+yww1Jd06SYgv3666+nusYCprtx7DTGMF2S46HQooop/Bw7fXf2Idekrg+2MV31hRdeKGWuOcZPTfPSK+sjqlfEU36hc5opa4yRQ4YMKWWmezKtV1OhrrnmmtTG9HL9LFpb0M6C1l8K41qdHQr7galajHt136PrjinXTM/W1C2u+ylTpqQ691QdZ8YFpr/pu6pFUkQ1TVPXM1PYaLPVtWvXUqZ0hymz3IM0phBawWjM4Zpj/2tqIu1DKLfQOuc77SB07BjzFidj0tjFtGOes3RfZwqk7k8R2T6N1mpMA5w2bVopc8+85JJLUp02T9qnr776atSh48FnqLMm4/zgPq/znW0822kKJ+2t6mQPETkF/uWXX05tlGGtvvrqpcx0fj1fRuR4yXW00047pbqmHlP2wDh82223pbrKshgXeA5XmMLM3wa6z1D6xX+r30s5FGU/mtbL9UkL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uYPPxH7S89DXv/711Ma97s9//nMpc21861vfaviMPJ8x1m6//falzPny0UcfpbrGy4iI3XffvZR57iMafxgXuPZ1rei6j6juddrHjz/+eMPPich7EM8onJf6vBERnTp1KuU777wztelZIiLHT74r+1TbeR7jM77++uul3Lp169TG8w9juO7Pffv2TW0TJkyIRugZZVHPuNlmm5Uy9zbdNyLy+bJVq1apTddcRI5HnJcrrLBCqs+fPz/Vdf/lOYrnFF0ffN6//vWvqa77FeMlY6uOnY5bRHXsuCZ1f+C87NGjR6qPHj06/lfxX0SNMcYYY4wxxjQV/xA1xhhjjDHGGNNU/EPUGGOMMcYYY0xTWWI0oqrBiojYddddU3327NmlTD3LVlttleqHHnpoql922WWlvNtuu6W25557LtU1J1t1VRERp556aqpfc801pay6toiqvkLbqVOljkNz4qmJoJaBuix9Hz4/9WmqOaC25IYbbogvCsdOtanTp09PbdTd/vznP0/1nj17lvLkyZNT2xZbbJHq+q7UoVDLM3To0FKm9o55+aoJ4bs98sgjqa5aPdUqRFT7W7UwEVnXQc0Z+0m1YdT5UIs3a9as+KJov+27776pbcqUKalObZKuO+obr7766lTfcccdS/mpp55KbdRuqvZ3yJAhqe3SSy9NdZ3jnAPUt1ADst5665Uy9VyqyY3IY8fvue6661JdNV2vvPJKauvatWuqDx8+vJS/zLhxzlJzSb2UzuMNNtggtfXu3TvVde1T60ttnuo1ueZOPvnkVB8/fvwinyeiqsEhGjM33XTT1Mb30XVHnTn1UtpPjJevvvpqqmsfUxfJOVAH1zbnhMY9anL1XoKIiIMPPriUqbP6xS9+keoaY6jf4ufqvORaZhygBlPfj/GSelONmdyLuVfrM7/44oupjXop3Tf33nvv1KbvFlHV/NXBdad7KNcc9YLrrrtuKR9wwAGpbebMmamu+yL7hfHyu9/9bikPHjw4tU2cODHVeU7R9+GZ5Z577kl1PafwTg/VrkVEbLLJJqXMeElN/fe///1Spv6be9IZZ5yR6g888EB8UXReduvWLbXpGTEi62W5bzNeah9Tm693U0RkTfHyyy+f2k466aRU13gZkc+9jCF1e90666yT2ni/gM4vagd5vrz88ssX+TwR1di65557prqeU2bMmBH/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7LLLWhv7LmMOPfSo5d9zzz2HMvV1XEeuueaaoUz/aXpGUkueUJNOTeJb3/rWocz4SPIa6THNcZkaohz7VePrz77iM6OPImN6rnX002bfpRae8YVep+kxuddee7W2Qw45pNVTz7hixYrWRh9v6k2pf0zyuVT1tZvnH1B3lX1J31Pq19O7lXsL7hfSP5Z9Rf9AepJm382fP7+18YyGiy++eChTH0g9YPoEM1bts88+rZ7xkutI7lH4vVVdr7+ytS7XN+43qVlMzSv3m/Q2TY0czxGhdjyf4b/+9a/Wxn0UdYncNyb0m8y1js+Qfs8J16tly5a1es479iP1jNTsZlxgXKMPbcKYQb9Sjr2E68rChQuHMvW8O++885zfe/PNN7c2atKpm893Be5DqMPNfSz3iDxrIHWf3Ftzb5rXwHMJeJ4D7z3nCr+XevCpeKlGVERERERERFZJfBEVERERERGRmbLKpOYyDYrXlekzTCNiugaPbM7jqzMVq6rqYx/7WKtnCiFT8Jim+cQTTwxlHv/NtIpMm9pwww1bWx4pXdVT5zIls6rbm1SNUw/yeOfdd9+9tTHV713vetdQZlodUytpBZBssskmrZ5HqvP47xdffLHVad+y9tprD2VasDBNMNNPMr2napx+m7/z85//vLXRjiPTFtg3TJPKo7eZIshUaB4P/u9//3so00Jjp512avW0JGK6D1NSMxXtF7/4RU2Rab1MR82Ur6px6nE+U8452otkKhqPTWcKedoTcK7wOb388stDmel6nCtMv8ojzJmex9TLtDbgXKF1TaaXsc9pxZAp5JxjU3OOaXNM02H6TKaBMY0r51xV1eLFi4cy5xzTw7LvmJ60wQYbtHraHjANmXP90UcfbfVM62K/0voo1wN+lpKJnN+05uDf5jPmnGOa11TfMaanVURV1UsvvTSUaWkyb968Vs94xL6i7UH2HVMIjz766Dk/yxhIKCPIVC2mpU3NX47pXF+rejxiDE8Lk6pu6bPGGmu0Nq5JtP95/PHHay44/jPNjumGtArKecdYxDTHTM3lHoXpt7lnYTrz73//+1anLViOA/YN91W5rjCtNG3Mqnrcy3Wuavx8M4Wc855xjbE3UzppYUJS1pTp71XjvssUSK75lA3sv//+Q5kyh09/+tOtns87LeWqxinjvJ+8RtqHMC054wTXbabmpl0X9wuU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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x11e2e0d90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: [0/200], Batch Num: [100/600]\n",
      "(2.0198584, 0.5187724)\n",
      "Discriminator Loss: 2.0199, Generator Loss: 0.5188\n",
      "D(x): 0.5076, D(G(z)): 0.6656\n"
     ]
    }
   ],
   "source": [
    "logger = Logger(model_name='VGAN', data_name='MNIST')\n",
    "\n",
    "for epoch in range(num_epochs):\n",
    "    for n_batch, (real_batch,_) in enumerate(data_loader):\n",
    "\n",
    "        # 1. Train Discriminator\n",
    "        real_data = Variable(images_to_vectors(real_batch))\n",
    "        if torch.cuda.is_available(): real_data = real_data.cuda()\n",
    "        # Generate fake data\n",
    "        fake_data = generator(noise(real_data.size(0))).detach()\n",
    "        # Train D\n",
    "        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,\n",
    "                                                                real_data, fake_data)\n",
    "\n",
    "        # 2. Train Generator\n",
    "        # Generate fake data\n",
    "        fake_data = generator(noise(real_batch.size(0)))\n",
    "        # Train G\n",
    "        g_error = train_generator(g_optimizer, fake_data)\n",
    "        # Log error\n",
    "        logger.log(d_error, g_error, epoch, n_batch, num_batches)\n",
    "\n",
    "        # Display Progress\n",
    "        if (n_batch) % 100 == 0:\n",
    "            display.clear_output(True)\n",
    "            # Display Images\n",
    "            test_images = vectors_to_images(generator(test_noise)).data.cpu()\n",
    "            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);\n",
    "            # Display status Logs\n",
    "            logger.display_status(\n",
    "                epoch, num_epochs, n_batch, num_batches,\n",
    "                d_error, g_error, d_pred_real, d_pred_fake\n",
    "            )\n",
    "        # Model Checkpoints\n",
    "        logger.save_models(generator, discriminator, epoch)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
