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
  def __init__(self, in_channels, out_channels, groups, act_fn, mask, stride, affine=True, deploy=False):
    super(MixedOp, self).__init__()
    self.deploy = deploy
    self.stride = stride
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.groups = groups
    #self.conv = ConvBnAct(in_channels, out_channels, 3, stride, affine, None)
    self._ops = nn.ModuleList()
    self.skip_connect = Identity(out_channels, groups, affine) if stride == 1 else None
    self.act_fn = act_fn

    for i, primitive in enumerate(PRIMITIVES):
        if mask[i] == 1:
            op = OPS[primitive](in_channels, out_channels, groups, stride, affine)
            self._ops.append(op)

  def forward(self, x):
    if not self.deploy:
        out = self.skip_connect(x) if self.skip_connect is not None else 0
        out += sum([op(x) for op in self._ops])
        out = self.act_fn(out)
    else:
        out = self.act_fn(self.conv(x))
    return out

  def fuse_weights(self):
    assert hasattr(self, '_ops'), "init ops list first!"
    w, b = self.skip_connect.fuse_weights() if self.skip_connect is not None else (0, 0)
    in_c, out_c, stride, groups = self.in_channels, self.out_channels, self.stride, self.groups
    w_l = [w]
    b_l = [b]
    self.conv = nn.Conv2d(in_c, out_c, 3, stride, 1, groups=groups)
    for i, m in enumerate(self._ops):
        w, b = m.fuse_weights()
        w_l.append(w)
        b_l.append(b)
    weights, bias = transII_addbranch(w_l, b_l)
    self.conv.weight.data.copy_(weights)
    self.conv.bias.data.copy_(bias)
    self.__delattr__('_ops')
    self.__delattr__('skip_connect')


class RepVGG(nn.Module):

    def __init__(self, num_blocks, mask, num_classes=1000, width_multiplier=None, criterion=None, override_groups_map=None):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4
        self._criterion = criterion
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.features = []
        self.features.append(MixedOp(3, self.in_planes, 1, nn.ReLU(True), mask[0], 2, True))
        self.override_groups_map = override_groups_map or dict()
        self.cur_layer_idx = 1
        self.mask = mask
        self.deploy = False
        self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.features = nn.Sequential(*self.features)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)
        self.dropout = nn.Dropout(0.1)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            if isinstance(cur_groups, str):
                cur_groups = planes
            self.features.append(MixedOp(self.in_planes, planes, cur_groups, nn.ReLU(True), self.mask[self.cur_layer_idx], stride, True))
            self.in_planes = planes
            self.cur_layer_idx += 1

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

    def fuse_weights(self):
        self.eval()
        for i, m in enumerate(self.features):
            m.fuse_weights()
            m.deploy = True

def create_RepVGG_A0(mask):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], mask=mask)

def create_RepVGG_A1(mask):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], mask=mask)

def create_RepVGG_B3(mask):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], mask=mask)

def create_RepVGG_B2g4(mask):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, mask=mask)


model_map = {
    'RepVGGA0': create_RepVGG_A0,
    'RepVGGA1': create_RepVGG_A1,
    'RepVGGB3': create_RepVGG_B3,
    'RepVGGB2g4': create_RepVGG_B2g4
}