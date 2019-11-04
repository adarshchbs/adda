import torch
from torchvision import datasets, transforms

import params

def get_mnist(train):

    pre_process = transforms.Compose( [transforms.Grayscale(3), transforms.ToTensor(), 
                                    transforms.Normalize( mean = params.dataset_mean,
                                     std = params.dataset_std ) ] )

    mnist_dataset = datasets.MNIST( 
                                    root = params.data_root,
                                    train = train,
                                    transform = pre_process,
                                    download = True
                                   )

    data_loader = torch.utils.data.DataLoader( 
                                                dataset = mnist_dataset,
                                                batch_size = params.batch_size,
                                                shuffle = True
                                            )

    
    return data_loader

# import numpy as np
# a = get_mnist(True)
# for i in a:
#     b = i[0]
#     c = b[0]
#     c = np.array(c)
#     print(c)