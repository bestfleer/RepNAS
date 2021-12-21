import torch.nn as nn
from .ddb_transforms import *

OPS = {
  'skip_connect': lambda in_channels, out_channels, groups, stride, affine:
  Identity(out_channels, groups, affine) if stride == 1 and in_channels == out_channels else None,
  'Conv_1x1': lambda in_channels, out_channels, groups, stride, affine:
  ConvBnAct(in_chs=in_channels, out_chs=out_channels, kernel_size=1, padding=0, stride=stride, affine=affine, groups=groups),
  'Conv_3x3': lambda in_channels, out_channels, groups, stride, affine:
  ConvBnAct(in_chs=in_channels, out_chs=out_channels, kernel_size=3, padding=1, stride=stride, affine=affine, groups=groups),
  'AvgPoolBN': lambda in_channels, out_channels, groups, stride, affine:
  AvgPoolBN(in_chs=in_channels, out_chs=out_channels, kernel_size=3, stride=stride, affine=affine, groups=groups),
  'Conv_1x1_3x3': lambda in_channels, out_channels, groups, stride, affine:
  Conv1x1_Convkxk(in_chs=in_channels, out_chs=out_channels, kernel_size=3, stride=stride, affine=affine, groups=groups),
  'Conv_1x3': lambda in_channels, out_channels, groups, stride, affine:
  ConvBnAct(in_chs=in_channels, out_chs=out_channels, kernel_size=(1, 3), padding=(0, 1), stride=stride, affine=affine, groups=groups),
  'Conv_3x1': lambda in_channels, out_channels, groups, stride, affine:
  ConvBnAct(in_chs=in_channels, out_chs=out_channels, kernel_size=(3, 1), padding=(1, 0), stride=stride, affine=affine, groups=groups),
}

class BNAndPadLayer(nn.Module):
  # copy from DDB, https://github.com/DingXiaoH/DiverseBranchBlock/blob/main/diversebranchblock.py
  def __init__(self,
               pad_pixels,
               num_features,
               eps=1e-5,
               momentum=0.1,
               affine=True,
               track_running_stats=True):
    super(BNAndPadLayer, self).__init__()
    self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
    self.pad_pixels = pad_pixels

  def forward(self, input):
    output = self.bn(input)
    if self.pad_pixels > 0:
      if self.bn.affine:
        pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(
          self.bn.running_var + self.bn.eps)
      else:
        pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
      output = F.pad(output, [self.pad_pixels] * 4)
      pad_values = pad_values.view(1, -1, 1, 1)
      output[:, :, 0:self.pad_pixels, :] = pad_values
      output[:, :, -self.pad_pixels:, :] = pad_values
      output[:, :, :, 0:self.pad_pixels] = pad_values
      output[:, :, :, -self.pad_pixels:] = pad_values
    return output

  @property
  def weight(self):
    return self.bn.weight

  @property
  def bias(self):
    return self.bn.bias

  @property
  def running_mean(self):
    return self.bn.running_mean

  @property
  def running_var(self):
    return self.bn.running_var

  @property
  def eps(self):
    return self.bn.eps


class Identity(nn.Module):

  def __init__(self, out_channels, groups, affine):
    super(Identity, self).__init__()
    self.bn = nn.BatchNorm2d(out_channels, affine=affine)
    self.groups = groups

  def forward(self, x, weights=None):
    x = self.bn(x)
    return x

  def copy_weights(self, weights):
    pass

  def fuse_weights(self):
    in_channels = self.bn.num_features//self.groups
    out_channels = self.bn.num_features
    weights = torch.zeros(out_channels, in_channels, 3, 3).to(self.bn.weight.device)
    for i in range(out_channels):
      weights[i, i%in_channels, 1, 1] = 1
    return transI_fusebn(weights, self.bn)

class ConvBnAct(nn.Module):
  def __init__(self, in_chs, out_chs, kernel_size, padding,
               stride=1, affine=True, act_fn=None, groups=1, bn_pad=False):
    super(ConvBnAct, self).__init__()
    assert stride in [1, 2]
    self.act_fn = act_fn
    self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
    if bn_pad:
        self.bn = BNAndPadLayer(1, out_chs, affine=affine)
    else:
        self.bn = nn.BatchNorm2d(out_chs, affine=affine)

  def forward(self, x, weights=None):
    if weights is None:
      x = self.bn(self.conv(x))
    else:
      if self.conv.kernel_size == (1, 1):
        weights = weights[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1)
      elif self.conv.kernel_size == (1, 3):
        weights = weights[:, :, 1, :].unsqueeze(2)
      elif self.conv.kernel_size == (3, 1):
        weights = weights[:, :, :, 1].unsqueeze(-1)
      elif self.conv.kernel_size == (3, 3):
        weights = weights
      x = F.conv2d(x, weights, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
      x = self.bn(x)
    if self.act_fn is not None:
      x = self.act_fn(x)
    return x

  def copy_weights(self, weights):
    if self.conv.kernel_size == (1, 1):
      weights = weights[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1)
    elif self.conv.kernel_size == (1, 3):
      weights = weights[:, :, 1, :].unsqueeze(2)
    elif self.conv.kernel_size == (3, 1):
      weights = weights[:, :, :, 1].unsqueeze(-1)
    elif self.conv.kernel_size == (3, 3):
      weights = weights
    self.conv.weight.data.copy_(weights)

  def fuse_weights(self):
    weights, bias = transI_fusebn(self.conv.weight.data, self.bn)
    if self.conv.kernel_size == (1, 1):
      weights = F.pad(weights, [1, 1, 1, 1])
    elif self.conv.kernel_size == (1, 3):
      weights = F.pad(weights, [0, 0, 1, 1])
    elif self.conv.kernel_size == (3, 1):
      weights = F.pad(weights, [1, 1, 0, 0])
    elif self.conv.kernel_size == (3, 3):
      weights = F.pad(weights, [0, 0, 0, 0])
    else:
      raise "not support kernel size = {}".format(self.conv.kernel_size)
    return weights, bias

class IdentityBasedConv1x1(nn.Conv2d):

  def __init__(self, channels, groups=1):
    super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1, stride=1,
                                               padding=0, groups=groups, bias=False)

    assert channels % groups == 0
    input_dim = channels // groups
    id_value = np.zeros((channels, input_dim, 1, 1))
    for i in range(channels):
      id_value[i, i % input_dim, 0, 0] = 1
    self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
    nn.init.zeros_(self.weight)

  def forward(self, input, weights=None):
    if weights is None:
      kernel = self.weight + self.id_tensor.to(self.weight.device)
    else:
      kernel = weights[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1) + self.id_tensor.to(self.weight.device)
    result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
    return result

  def copy_weights(self, weights):
    self.weight.data.copy_(weights[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1))

class AvgPoolBN(nn.Module):
  def __init__(self, in_chs, out_chs, kernel_size,
               stride=1, affine=True, groups=1):
    super(AvgPoolBN, self).__init__()
    self.conv1 = ConvBnAct(in_chs=in_chs, out_chs=out_chs, kernel_size=1,
                           padding=0, stride=1, affine=affine, groups=groups, bn_pad=True)
    self.avgpool = nn.AvgPool2d(kernel_size, stride, 0)
    self.bn2 = nn.BatchNorm2d(out_chs, affine=affine)

  def forward(self, x, weights=None):
    x = self.conv1(x, weights)
    x = self.bn2(self.avgpool(x))
    return x

  def copy_weights(self, weights):
    self.conv1.copy_weights(weights)

  def fuse_weights(self):
    weights_1x1, bias_1x1 = transI_fusebn(self.conv1.conv.weight, self.conv1.bn)
    weights_avg = transV_avg(self.conv1.conv.out_channels, 3, self.conv1.conv.groups)
    weights_avg, bias_avg = transI_fusebn(weights_avg, self.bn2)
    return transIII_1x1_kxk(weights_1x1, bias_1x1, weights_avg, bias_avg, self.conv1.conv.groups)

class Conv1x1_Convkxk(nn.Module):
  def __init__(self, in_chs, out_chs, kernel_size,
               stride=1, affine=True, groups=1):
    super(Conv1x1_Convkxk, self).__init__()
    conv_pad = kernel_size // 2 if stride != 1 else 0
    if stride == 1:
      self.conv1 = IdentityBasedConv1x1(in_chs, groups)
      self.bn1 = BNAndPadLayer(1, in_chs, affine=affine)
    self.conv2 = ConvBnAct(in_chs=in_chs, out_chs=out_chs, kernel_size=3,
                           padding=conv_pad, stride=stride, affine=affine, groups=groups)

  def forward(self, x, weights=None):
    if hasattr(self, 'conv1'):
      x = self.bn1(self.conv1(x, weights))
    x = self.conv2(x, weights)
    return x

  def copy_weights(self, weights):
    if hasattr(self, 'conv1'):
      self.conv1.copy_weights(weights)
    self.conv2.copy_weights(weights)

  def fuse_weights(self):
    weights_kxk, bias_kxk = transI_fusebn(self.conv2.conv.weight, self.conv2.bn)
    if hasattr(self, 'conv1'):
      weights_1x1, bias_1x1 = transI_fusebn(self.conv1.weight+self.conv1.id_tensor, self.bn1)
      return transIII_1x1_kxk(weights_1x1, bias_1x1, weights_kxk, bias_kxk, self.conv1.groups)
    else:
      return weights_kxk, bias_kxk