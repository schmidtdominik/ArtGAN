import torch
import torch.nn as nn


class BatchStdConcat(nn.Module):
    """
    Add std to last layer group of disc to improve variance
    from https://github.com/nvnbny/progressive_growing_of_gans/blob/master/modelUtils.py
    """

    def __init__(self, groupSize=4):
        super().__init__()
        self.groupSize = 4

    def forward(self, x):
        shape = list(x.size())  # NCHW - Initial size
        xStd = x.view(self.groupSize, -1, shape[1], shape[2], shape[3])  # GMCHW - split batch as groups of 4
        xStd -= torch.mean(xStd, dim=0, keepdim=True)  # GMCHW - Subract mean of shape 1MCHW
        xStd = torch.mean(xStd ** 2, dim=0, keepdim=False)  # MCHW - Take mean of squares
        xStd = (xStd + 1e-08) ** 0.5  # MCHW - Take std
        xStd = torch.mean(xStd.view(int(shape[0] / self.groupSize), -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
        # M111 - Take mean across CHW
        xStd = xStd.repeat(self.groupSize, 1, shape[2], shape[3])  # N1HW - Expand to same shape as x with one channel
        return torch.cat([x, xStd], 1)

    def __repr__(self):
        return self.__class__.__name__ + '(Group Size = %s)' % (self.groupSize)


class PixelNormalization(nn.Module):
    """
    This is the per pixel normalization layer. This will devide each x, y by channel root mean square
    from: https://github.com/nvnbny/progressive_growing_of_gans/blob/master/modelUtils.py
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8) ** 0.5

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)


class PrintLayer(nn.Module):
    def __init__(self, *string_args):
        super(PrintLayer, self).__init__()
        self.string = ', '.join([str(k) for k in string_args])

    def forward(self, x):
        print('PL:', self.string, x.shape)
        return x

def get_gen_block(input_depth, res):
    print(input_depth, res)

    block = nn.Sequential(
        #PrintLayer('block', input_depth, res),
        nn.Conv2d(input_depth, input_depth//2, 3, 1, 1, bias=True),  # TODO: bias=False?
        nn.LeakyReLU(0.2, inplace=True),
        # TODO: pixelnorm
        nn.Conv2d(input_depth//2, input_depth//2, 3, 1, 1, bias=True),  # TODO: bias=False?
        nn.LeakyReLU(0.2, inplace=True),
        # TODO: pixelnorm
    )
    to_rgb_layer = nn.Conv2d(input_depth//2, 3, 1, 1, 0, bias=True)

    return block, to_rgb_layer

class Generator(nn.Module):
    def __init__(self, z_size, depth_modifier, resolution_levels, depth_levels):
        super(Generator, self).__init__()
        self.transition_phase = None
        self.alpha = None
        self.current_level = None

        self.blocks = torch.nn.ModuleList()
        self.to_rgb_layers = torch.nn.ModuleList()

        self.blocks.append(nn.Sequential(
            # TODO: pixelnorm
            nn.Conv2d(z_size//(4**2), depth_modifier * depth_levels[0], 3, 1, 1, bias=True),  # TODO: bias=False?
            nn.LeakyReLU(0.2, inplace=True),
            # TODO: pixelnorm
            nn.Conv2d(depth_modifier * depth_levels[0], depth_modifier * depth_levels[0], 3, 1, 1, bias=True),  # TODO: bias=False?
            nn.LeakyReLU(0.2, inplace=True),
            # TODO: pixelnorm
        ))
        self.to_rgb_layers.append(nn.Conv2d(depth_modifier * depth_levels[0], 3, 1, 1, 0, bias=True))

        for i in range(1, len(depth_levels)):
            block, to_rgb = get_gen_block(depth_levels[i-1]*depth_modifier, resolution_levels[i-1]) # res is input resolution for layer, INCLUDING upsampling
            self.blocks.append(block)
            self.to_rgb_layers.append(to_rgb)

        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    # 4x4 → 8x8T → 8x8 → 16x16T → 16x16 → ...

    def forward(self, input):
        input = input.view(input.shape[0], -1, 4, 4)
        #print('reshaped z', input.shape)
        intermediate_outputs = []
        for i in range(0, self.current_level+1):
            if i != 0:
                input = self.upsample(input)
                #print('upsampled', input.shape, i)

            input = self.blocks[i](input) # res is input resolution for layer, INCLUDING upsampling
            #print('after block', input.shape, i)
            intermediate_outputs.append(input)

        hr_out = self.to_rgb_layers[self.current_level](intermediate_outputs[-1])
        #print('hr_out', hr_out.shape)

        if not self.transition_phase:
            return hr_out
        else:
            lr_out = self.to_rgb_layers[self.current_level-1](intermediate_outputs[-2])
            #print('lr_out', lr_out.shape)

            lr_out = self.upsample(lr_out)
            #print('lr_out upsampled', lr_out.shape)

            return self.alpha*hr_out + (1-self.alpha)*lr_out


# GEN: nz 10 8 4 2 1 nc | DIS: nc 1 2 4 8 10 1

def get_disc_block(input_depth, res):
    from_rgb_layer = nn.Sequential(
        nn.Conv2d(3, input_depth, 1, 1, 0, bias=True),
        nn.LeakyReLU(0.2, inplace=True),
    )

    block = nn.Sequential(
        #PrintLayer('discblock', input_depth, res),
        nn.Conv2d(input_depth, input_depth, 3, 1, 1, bias=True),  # TODO: bias=False?
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(input_depth, input_depth*2, 3, 1, 1, bias=True),  # TODO: bias=False?
        nn.LeakyReLU(0.2, inplace=True),
    )

    return block, from_rgb_layer

class Discriminator(nn.Module):
    def __init__(self, z_size, depth_modifier, resolution_levels, depth_levels):
        super(Discriminator, self).__init__()
        self.transition_phase = None
        self.alpha = None
        self.current_level = None
        self.depth_levels = depth_levels

        self.blocks = torch.nn.ModuleList()
        self.from_rgb_layers = torch.nn.ModuleList()

        for dl, res in list(zip(reversed(depth_levels), reversed(resolution_levels)))[:-1]:
            block, from_rgb = get_disc_block(dl*depth_modifier, res)
            self.blocks.append(block)
            self.from_rgb_layers.append(from_rgb)

        self.from_rgb_layers.append(nn.Conv2d(3, depth_modifier * depth_levels[0], 1, 1, 0, bias=True))
        self.blocks.append(nn.Sequential(
            # TODO: Minibatch std
            nn.Conv2d(depth_modifier * depth_levels[0], depth_modifier * depth_levels[0], 3, 1, 1, bias=True),  # TODO: bias=False?
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(depth_modifier * depth_levels[0], 1, 4, 1, 0, bias=True),  # TODO: bias=False?
            nn.LeakyReLU(0.2, inplace=True),
        ))

        self.downsample = nn.AvgPool2d(2, stride=2, padding=0)
        # 4x4 → 8x8T → 8x8 → 16x16T → 16x16 → ...

    def forward(self, input):
        array_index = (len(self.depth_levels)-1)-self.current_level
        if not self.transition_phase:
            #print('INPUT', input.shape)
            input = self.from_rgb_layers[array_index](input)
            #print('INPUT FROM RGB', input.shape)
            for i, l in enumerate(self.blocks[array_index:]):
                input = l(input)
                #print('APPLIED BLOCK', input.shape)
                if i != len(self.blocks[array_index:])-1:
                    input = self.downsample(input)
                    #print('DOWNSAMPLED', input.shape)
            return input
        else:
            # hr
            hr = self.from_rgb_layers[array_index](input)
            #print('HR FROM RGB', hr.shape)
            hr = self.blocks[array_index](hr)
            #print('HR APPLIED BLOCK', hr.shape)
            hr = self.downsample(hr)
            #print('HR DOWNSAMPLED', hr.shape)

            # lr
            lr = self.downsample(input)
            #print('LR DOWNSAMPLED', lr.shape)
            lr = self.from_rgb_layers[array_index+1](lr)
            #print('LR FROM RGB', lr.shape)

            merged = self.alpha*hr+(1-self.alpha)*lr
            #print('MERGE', merged.shape)
            for i, l in enumerate(self.blocks[array_index+1:]):
                merged = l(merged)
                if i != len(self.blocks[array_index+1:])-1:
                    merged = self.downsample(merged)
            return merged