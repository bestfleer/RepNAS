from .operations import *

PRIMITIVES = ['Conv_1x1',
              'AvgPoolBN',
              'Conv_3x3',
              'Conv_1x1_3x3',
              'Conv_1x3',
              'Conv_3x1',
            ]

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

class MixedOp(nn.Module):
  def __init__(self, in_channels, out_channels, groups, act_fn, stride, affine=True, deploy=False):
    super(MixedOp, self).__init__()
    self.deploy = deploy
    self.fixed = True
    self.conv = ConvBnAct(in_channels, out_channels, 3, 1, stride, groups=groups)
    self.skip_connect = Identity(out_channels, groups, True) if stride == 1 else None
    self._ops = nn.ModuleList()
    self.stride = stride
    self.groups = groups
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.act_fn = act_fn
    for primitive in PRIMITIVES:
      op = OPS[primitive](in_channels, out_channels, groups, stride, affine)
      self._ops.append(op)

  def forward(self, x, rank=None, mask=None):
    if not self.deploy:
        weight = self.conv.conv.weight if not self.fixed else None
        out = self.skip_connect(x) if self.skip_connect is not None else 0
        for i in range(len(mask)):
            if mask[i] != 0:
                out += rank[i] * self._ops[i](x, weight)
        if isinstance(out, int):
            out = self._ops[2](x, weight)
        out = self.act_fn(out)
    else:
        out = self.act_fn(self.conv(x))
    return out

  def copy_weights(self):
    weight = self.conv.conv.weight.data
    for i, m in enumerate(self._ops):
        m.copy_weights(weight.clone())
    self.fixed = True

  def fuse_weights(self, mask):
    assert hasattr(self, '_ops'), "init ops list first!"
    w, b = self.skip_connect.fuse_weights()
    in_c, out_c, stride, padding, groups = self.conv.conv.in_channels, self.conv.conv.out_channels, \
                                           self.conv.conv.stride, self.conv.conv.padding,\
                                           self.conv.conv.groups
    w_l = [w]
    b_l = [b]
    self.conv = nn.Conv2d(in_c, out_c, 3, stride, padding, groups=groups)
    for i, m in enumerate(self._ops):
        if mask[i].item() == 0 or m is None:
            continue
        w, b = m.fuse_weights()
        w_l.append(w)
        b_l.append(b)
    weights, bias = transII_addbranch(w_l, b_l)
    self.conv.weight.data.copy_(weights)
    self.conv.bias.data.copy_(bias)
    self.__delattr__('_ops')
    self.__delattr__('skip_connect')

class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, temp=0.5):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4
        self.temp = temp
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.features = nn.ModuleList()
        self.features.append(MixedOp(3, self.in_planes, 1, nn.ReLU(True), 2, True))
        self.override_groups_map = override_groups_map or dict()
        self.cur_layer_idx = 1
        self.no_identity_layer_indx = [0]
        self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)
        self.dropout = nn.Dropout(0.1)
        self._init_alphas()
        self.random_sample = False

    def _init_alphas(self):
        mixop_num = len(self.features)
        self.alphas = torch.zeros(mixop_num, len(PRIMITIVES))
        #self.alphas[self.no_identity_layer_indx, 2] = 10000
        self.alphas = nn.Parameter(self.alphas)
        self.original_ops = self.alphas.numel()
        self.fixed_path = None


    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        self.no_identity_layer_indx.append(self.cur_layer_idx)
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            if isinstance(cur_groups, str):
                cur_groups = planes
            self.features.append(MixedOp(self.in_planes, planes, cur_groups, nn.ReLU(True), stride, True))
            self.in_planes = planes
            self.cur_layer_idx += 1

    def forward(self, x):
        if hasattr(self, 'alphas'):
            rank, mask = self.prune()
            for i, block in enumerate(self.features):
                x = block(x, rank[i], mask[i])
            x = self.gap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            out = self.linear(x)
            return out, rank
        else:
            x = self.features(x)
            x = self.gap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            out = self.linear(x)
            return out

    def prune(self):
        rank = self._get_perturbation()
        if self.fixed_path is not None:
            mask = self.fixed_path
            rank = self.fixed_path.float()
        else:
            masked_index = rank.view(-1).sort(descending=True)[1][:int(self.original_ops * self.constraint)]
            mask = torch.zeros_like(rank.view(-1), memory_format=torch.legacy_contiguous_format).scatter(0, masked_index, 1).reshape(rank.size())
            #mask = (rank >= 0).float()
            rank = mask + rank.sigmoid() - rank.sigmoid().detach()
        return rank, mask

    def fixed_mask(self):
        self.eval()
        masked_index = self.alphas.view(-1).sort(descending=True)[1][:int(self.original_ops * self.constraint)]
        mask = torch.zeros_like(self.alphas.view(-1)).scatter(0, masked_index, 1).reshape(self.alphas.size())
        self.fixed_path = mask
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
        #self.copy_weights()

    def copy_weights(self):
        for i, m in enumerate(self.features):
            m.copy_weights()

    def fuse_weights(self):
        self.eval()
        if self.fixed_path is None:
            raise Exception("search the fixed path first!")
        mask = self.fixed_path
        for i, m in enumerate(self.features):
            m.fuse_weights(mask[i])
        self.features = nn.Sequential(*self.features)
        self.__delattr__('alphas')

    def _get_perturbation(self):
        if self.training:
            perturbation1 = -torch.log(-torch.log(torch.rand(self.alphas.size()))).to(self.alphas.device)
            perturbation2 = -torch.log(-torch.log(torch.rand(self.alphas.size()))).to(self.alphas.device)
        else:
            perturbation1 = 0
            perturbation2 = 0
        rank = self.alphas + perturbation1 - perturbation2
        return rank

def create_RepVGG_A0():
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None)

def create_RepVGG_A1():
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None)

def create_RepVGG_A2():
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None)

def create_RepVGG_B1():
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None)

def create_RepVGG_B2():
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None)

def create_RepVGG_B3():
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None)


model_map = {
    'RepVGGA0': create_RepVGG_A0,
    'RepVGGA1': create_RepVGG_A1,
    'RepVGGA2': create_RepVGG_A2,
    'RepVGGB1': create_RepVGG_B1,
    'RepVGGB2': create_RepVGG_B2,
    'RepVGGB3': create_RepVGG_B3,

}
