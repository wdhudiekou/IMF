import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

shape = [256, 256]
# shape = [256, 320]
# shape = [240, 320]

class ConvBnLeakyRelu2d(nn.Module):
    '''Conv2d + BN + LeakyReLU'''
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)

class TransConvBnLeakyRelu2d(nn.Module):
    '''ConvTranspose2d + BN + LeakyReLU'''
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, dilation=1, groups=1):
        super(TransConvBnLeakyRelu2d, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.transconv(x)), negative_slope=0.1)

class ConvSigmoid(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvSigmoid, self).__init__()
        self.conv     = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.sigmoid  = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv(x))

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

def predict_flow(in_channels, dim=2, kernel_size=3, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, dim, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, volsize, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        size = volsize
        gpu_use = True
        vectors = [ torch.arange(0, s) for s in size ]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids) # y, x, z
        grid = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...].clone() / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode='zeros', align_corners=True), new_locs

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """
    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)[0]
        return vec


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True)):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=8, kernel_size=3):
        super(Encoder, self).__init__()

        self.encoder_level_3 = [ConvBnLeakyRelu2d(in_channels, out_channels, kernel_size, stride=2) for _ in range(1)] # 3-> 8
        self.encoder_level_2 = [
            ConvBnLeakyRelu2d(out_channels, 2 * out_channels, kernel_size, stride=2), # 8 -> 16
            ResBlock(conv, 2 * out_channels, kernel_size, bias=True, bn=False, act=nn.LeakyReLU(0.1))
        ]
        self.encoder_level_1 = [
            ConvBnLeakyRelu2d(2 * out_channels, 4 * out_channels, kernel_size, stride=2),  # 16 -> 32
            ResBlock(conv, 4 * out_channels, kernel_size, bias=True, bn=False, act=nn.LeakyReLU(0.1))
        ]
        self.encoder_level_0 = [
            ConvBnLeakyRelu2d(4 * out_channels, 8 * out_channels, kernel_size, stride=2),  # 32 -> 64
            ResBlock(conv, 8 * out_channels, kernel_size, bias=True, bn=False, act=nn.LeakyReLU(0.1))
        ]

        self.encoder_level_0 = nn.Sequential(*self.encoder_level_0)
        self.encoder_level_1 = nn.Sequential(*self.encoder_level_1)
        self.encoder_level_2 = nn.Sequential(*self.encoder_level_2)
        self.encoder_level_3 = nn.Sequential(*self.encoder_level_3)

    def forward(self, x):
        enc3 = self.encoder_level_3(x)    # torch.Size([2, 8, 256, 256])
        enc2 = self.encoder_level_2(enc3) # torch.Size([2, 16, 128, 128])
        enc1 = self.encoder_level_1(enc2) # torch.Size([2, 32, 64, 64])
        enc0 = self.encoder_level_0(enc1) # torch.Size([2, 64, 32, 32])

        return [enc3, enc2, enc1, enc0]


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.avg_fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias)
        )
        # global max pooling: feature --> point
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.max_fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_fc(self.avg_pool(x))
        max_out = self.max_fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return out

class PFF(nn.Module):
    def __init__(self, in_channels):
        super(PFF, self).__init__()
        self.channels = in_channels

        self.downconv_1 = nn.Sequential(conv(3 * self.channels, self.channels, kernel_size=1),
                                        nn.ReLU(inplace=True))
        self.plainconv  = nn.Sequential(conv(self.channels, self.channels, kernel_size=3),
                                        nn.ReLU(inplace=True))

        self.downconv_2 = nn.Sequential(conv(self.channels, 3, kernel_size=3),
                                        nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=-1)

        self.ca_layer = ChannelAttention(3 * self.channels)


    def forward(self, dec, mov_warp, fix_enc):
        dec_feat_in = torch.cat([dec, mov_warp, fix_enc], dim=1) # [B, 3*C, H, W] torch.Size([2, 96, 64, 64])

        dec_feat  = self.plainconv(self.downconv_1(dec_feat_in)) # [B, C, H, W] torch.Size([2, 32, 64, 64])
        down_feat_in = self.downconv_2(dec_feat) # [B, 3, H, W] torch.Size([2, 3, 64, 64])
        down_feat    = down_feat_in.view(down_feat_in.shape[0], down_feat_in.shape[1], -1) # [B, 3, H * W] torch.Size([2, 3, 4096])

        weight_maps = self.softmax(down_feat) # [B, 3, H * W] torch.Size([2, 3, 4096])
        weight_maps = weight_maps.view(down_feat_in.shape[0], down_feat_in.shape[1], down_feat_in.shape[2], down_feat_in.shape[3]) # [B, 3, H, W] torch.Size([2, 3, 64, 64])
        spatial_dec      = dec * weight_maps[:, :1, :, :]
        spatial_mov_warp = mov_warp * weight_maps[:, 1:2, :, :]
        spatial_fix_enc  = fix_enc * weight_maps[:, 2:, :, :]
        spatial_fms = torch.cat([spatial_dec, spatial_mov_warp, spatial_fix_enc], dim=1) # torch.Size([2, 96, 64, 64])

        channel_wise = self.ca_layer(spatial_fms) # torch.Size([2, 96, 1, 1])
        output = channel_wise * spatial_fms # torch.Size([2, 96, 64, 64])

        return output

class DFF(nn.Module):
    def __init__(self, in_channels=2, list_num=4):
        super(DFF, self).__init__()
        self.channels = in_channels
        self.num = list_num
        self.exp = 16
        self.step = 7
        self.conv_1 = nn.Sequential(conv(self.num * self.channels, self.exp * self.channels, kernel_size=3, stride=1),
                                    nn.ReLU(inplace=True)) # num*2 -> 16*2
        self.conv_2 = nn.Sequential(conv(self.exp * self.channels, self.exp * self.channels, kernel_size=3, stride=1),
                                    nn.ReLU(inplace=True))# 16*2 -> 16*2

        self.convsig = [ConvSigmoid(self.exp * self.channels, 2, kernel_size=3, stride=1) for _ in range(self.num)]
        self.convsig = nn.Sequential(*self.convsig)

    def forward(self, predict_flows): # torch.Size([2, 2, 32, 32])

        pred_cache = []
        for i, flow in enumerate(predict_flows):
            pred_cache.append(F.interpolate(flow, scale_factor=(2**(self.num-i), 2**(self.num-i)), mode='bilinear', align_corners=True))

        pred_cat = torch.cat(pred_cache, dim=1) # torch.Size([2, 2, 64, 64])
        weights_cat = self.conv_2(self.conv_1(pred_cat)) # torch.Size([2, 32, 64, 64])

        for i, flow in enumerate(pred_cache):
            weight_map = self.convsig[i](weights_cat) # torch.Size([2, 2, 64, 64])
            pred_cache[i] = flow * weight_map
            if i==0:
                progress_field = pred_cache[i] # torch.Size([2, 2, 64, 64])
            else:
                progress_field = progress_field + pred_cache[i] #
        self.integrate = VecInt([progress_field.shape[2], progress_field.shape[3]], self.step)
        progress_field = self.integrate(progress_field) # torch.Size([2, 2, 64, 64])

        return progress_field


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class MPDRNet(nn.Module):

    def __init__(self, init_channels=16):
        super(MPDRNet, self).__init__()
        self.in_channels = 1
        self.channels = init_channels
        self.kernel_size = 3

        # TODO: Encoder of the fixed image
        self.fix_encoder = Encoder(self.in_channels, self.channels, self.kernel_size)

        # TODO: Encoder of the moving image
        self.mov_encoder = Encoder(self.in_channels, self.channels, self.kernel_size)

        # TODO: Bottle neck
        self.bottle_1 = ConvBnLeakyRelu2d(16 * self.channels, 8 * self.channels, self.kernel_size)
        self.bottle_2 = ConvBnLeakyRelu2d(8  * self.channels, 8 * self.channels, self.kernel_size)

        # TODO: TransposeConv2d of the Decoder
        self.upsample0 = TransConvBnLeakyRelu2d(8 * self.channels, 4 * self.channels, 4, 2, 1)
        self.upsample1 = TransConvBnLeakyRelu2d(12 * self.channels, 2 * self.channels, 4, 2, 1)
        self.upsample2 = TransConvBnLeakyRelu2d(6 * self.channels, self.channels, 4, 2, 1)

        # TODO: Predict flow-0/1/2/3
        self.predict_flow0 = predict_flow(8 * self.channels, 2)
        self.predict_flow1 = predict_flow(12 * self.channels, 2)
        self.predict_flow2 = predict_flow(6 * self.channels, 2)
        self.predict_flow3 = predict_flow(3 * self.channels, 2)

        # TODO: Deformation Field Fusion (DFF)
        self.DFF_1 = DFF(list_num=1)
        self.DFF_2 = DFF(list_num=2)
        self.DFF_3 = DFF(list_num=3)
        self.DFF_4 = DFF(list_num=4)

        # TODO: Progressive Feature Fine (PFF)
        self.PFF_1 = PFF(4 * self.channels)
        self.PFF_2 = PFF(2 * self.channels)
        self.PFF_3 = PFF(self.channels)

        self.resize = ResizeTransform(2, 2)

    def load_state_dict(self, state_dict, strict = False):
        state_dict.pop('spatial_transform_f1.grid')
        state_dict.pop('spatial_transform_f2.grid')
        super().load_state_dict(state_dict, strict)

    def forward(self, fix_img, mov_img, shape=shape):
        """
        source images pair ([Tensor(batch, c, height, weight)] or Tensor(batch, c, height, weight))
        :return: deformation fields
        """
        # TODO: Warping phi-1/2/3 and mov_enc-1/2/3/
        if shape is not None:
            down_shape1 = [int(d / 8) for d in shape]
            down_shape2 = [int(d / 4) for d in shape]
            down_shape3 = [int(d / 2) for d in shape]
            self.spatial_transform_f1 = SpatialTransformer(volsize=down_shape1)
            self.spatial_transform_f2 = SpatialTransformer(volsize=down_shape2)
            self.spatial_transform_f3 = SpatialTransformer(volsize=down_shape3)
            self.spatial_transform_im = SpatialTransformer(volsize=shape)

        [mov_enc3, mov_enc2, mov_enc1, mov_enc0] = self.mov_encoder(mov_img)
        [fix_enc3, fix_enc2, fix_enc1, fix_enc0] = self.fix_encoder(fix_img)

        bottle_feat = torch.cat([mov_enc0, fix_enc0], dim=1)
        dec0 = self.bottle_2(self.bottle_1(bottle_feat))

        predict_flows = []

        # predict flow-0
        flow0 = self.predict_flow0(dec0)
        predict_flows.append(flow0)

        # warping moving feature-1
        up_dec0 = self.upsample0(dec0)
        phi_1   = self.DFF_1(predict_flows)
        warped_1, _ = self.spatial_transform_f1(mov_enc1, phi_1)
        dec1 = self.PFF_1(up_dec0, warped_1, fix_enc1)
        # predict flow-1
        flow1 = self.predict_flow1(dec1)
        predict_flows.append(flow1)

        # warping moving feature-2
        up_dec1 = self.upsample1(dec1)
        phi_2   = self.DFF_2(predict_flows)
        warped_2, _ = self.spatial_transform_f2(mov_enc2, phi_2)
        dec2 = self.PFF_2(up_dec1, warped_2, fix_enc2)
        # predict flow-2
        flow2 = self.predict_flow2(dec2)
        predict_flows.append(flow2)

        # warping moving feature-3
        up_dec2 = self.upsample2(dec2)
        phi_3   = self.DFF_3(predict_flows)
        warped_3, _ = self.spatial_transform_f3(mov_enc3, phi_3)
        dec3 = self.PFF_3(up_dec2, warped_3, fix_enc3)
        # predict flow-2
        flow3 = self.predict_flow3(dec3)
        predict_flows.append(flow3)

        # warping moving image
        phi_4      = self.DFF_4(predict_flows)
        final_flow = phi_4
        warped_mov, disp_pre = self.spatial_transform_im(mov_img, final_flow)
        warped_fix, _ = self.spatial_transform_im(fix_img, (-final_flow))

        return warped_mov, warped_fix, final_flow, phi_3, phi_2, disp_pre


class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """
    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out

class ConstuctmatrixLayer(nn.Module):
    def __init__(self):
        super(ConstuctmatrixLayer, self).__init__()

    def forward(self, angle, scale_x, scale_y, center_x, center_y):
        theta, matrix = construct_M(angle, scale_x, scale_y, center_x, center_y)
        return theta, matrix

def construct_M(angle, scale_x, scale_y, center_x, center_y):
    alpha = torch.cos(angle)
    beta = torch.sin(angle)
    tx = center_x
    ty = center_y
    tmp0 = torch.cat((scale_x * alpha, beta), 1)
    tmp1 = torch.cat((-beta, scale_y * alpha), 1)
    theta = torch.cat((tmp0, tmp1), 0)
    t = torch.cat((tx, ty), 0)
    matrix = torch.cat((theta, t), 1)
    return theta, matrix



def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()

if __name__ == '__main__':
    #
    model = MPDRNet().cuda()
    a = torch.randn(2, 1, 256, 256).cuda()
    b = torch.randn(2, 1, 256, 256).cuda()
    warped_mov, warped_fix, final_flow, _, _, disp_pre = model(a,b)
    print(warped_mov.shape, warped_fix.shape, final_flow.shape)
    model.eval()
    print("Params(M): %.2f" % (params_count(model) / (1000 ** 2)))

