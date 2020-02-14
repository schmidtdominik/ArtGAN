import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

def get_dataset(name, image_size, batch_size, workers):
    if name == 'celeba':
        celeba_dataset = dset.ImageFolder(root='./celeba',
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        celeba_dataloader = torch.utils.data.DataLoader(celeba_dataset, batch_size=batch_size,
                                                        shuffle=True, num_workers=workers)
        return celeba_dataloader
    elif name == 'art':
        art_dataset = dset.ImageFolder(root='./data/art_cropped_train',
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.RandomCrop(image_size),
                                       transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))
        art_dataloader = torch.utils.data.DataLoader(art_dataset, batch_size=batch_size,
                                                     shuffle=True, num_workers=workers)
        return art_dataloader

