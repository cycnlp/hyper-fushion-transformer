
class Conv(nn.Module):
    # def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)
# x=SpatialAttention()
# y=torch.randn(2,3,96,96)
# print(x(y).shape)
#
# print("-----97")
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        # self.bn = BatchNorm2d(out_chan, activation='none')
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def initialize_weights(*models):
   for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


# x=DEPTHWISECONV(96,96)
# y=torch.randn(2,96,1,1)
# print(x(y).shape)





class crossBlock(nn.Module):
    def __init__(self, channels):
        super(crossBlock, self).__init__()

        # self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        width=channels
        self.conv3 =ConvBNReLU(channels,channels)
        self.conv4 =ConvBNReLU(3*channels,channels//2)

        self.dilation2 = nn.Sequential(SeparableConv2d(width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))


        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        # x1 = self.agv(x)
        # x1 = self.con3(x)
        input=x

        x1 = self.conv3(x)
        # x3 = self.relu3(x3)
        x2=self.dilation2(x)+x
        x3=self.dilation3(x)+x
        x=torch.cat([ x1,x3, x2],dim=1)
        out = self.conv4(x)

        return out
class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

# 修改的代码
class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels , 1)
        self.norm1 = nn.BatchNorm2d(in_channels )

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels , in_channels , kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels )

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels , n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x
class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)#b 512 1 1
        y=y.view(b, c) #2 512
        y = self.fc(y) #2 512
        y=y.view(b, c, 1, 1)#b 512 1 1
        return x * y
class net1(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.resnet = resnet
        self.swin = swin

  
        self.conv0 = Conv(64, 96, bn=True, relu=True)
        self.conv1 = Conv(128, 192, 3, bn=True, relu=True)
        self.conv2 = Conv(256, 384, 3, bn=True, relu=True)
        self.conv3 = Conv(512, 768, 3, bn=True, relu=True)
        # self.spp = SPPblock(768)


        self.resblock0 = BasicBlock(96, 96)
        self.resblock1 = BasicBlock(192, 192)
        self.resblock2 = BasicBlock(384, 384)
        self.resblock3 = BasicBlock(768, 768)
        # self.conv1=nn.Conv2d(192,96,kernel_size=3,stride=1,padding=1)

        filters = [96, 192, 384, 768]

        self.decoder3 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder2 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder1 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder0 = DecoderBlockLinkNet(filters[0], filters[0])


        self.cro2=crossBlock(384*2)
        self.cro1=crossBlock(192*2)
        self.cro0=crossBlock(96*2)

        self.final2 = nn.Conv2d(384, num_class, kernel_size=1)
        self.final1 = nn.Conv2d(192, num_class, kernel_size=1)
        self.final = nn.Conv2d(96, num_class, kernel_size=1)


        self.avg=nn.AdaptiveAvgPool2d(1)
        # self.atchan0=DEPTHWISECONV(96,96)
        # self.atchan1=DEPTHWISECONV(192,192)
        # self.atchan2=DEPTHWISECONV(384,384)
        # self.atchan3=DEPTHWISECONV(768,768)
        #
        self.se0 = se_block(96)
        self.se1 = se_block(192)
        self.se2 = se_block(384)
        self.se3 = se_block(768)


        self.spation=SpatialAttention()

        # self.convbian=nn.Conv2d(288,1,kernel_size=1)
        # self.sigmod=nn.Sigmoid()
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x, gts=None,criterion=None):
        input = x
        x_size=x.size()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_x = self.resnet.maxpool(x)  # bs, 64, 64, 64
        # ---- low-level features ----
        x0 = self.resnet.layer1(x_x)  # bs, 64, 64, 64

        x_ = x0
        x_ = self.drop(x_)
        x1 = self.resnet.layer2(x_)  # bs 128 32 32
        x1 = self.drop(x1)

        x2 = self.resnet.layer3(x1)  # bs,  256 16 16
        x2 = self.drop(x2)

        x3 = self.resnet.layer4(x2)  # bs  512 8 8
        x3 = self.drop(x3)

        self.relu=nn.ReLU(inplace=True)

        swin_x, h, w = self.swin.patch_embed(input)
        # swin_x=self.swin.LayerNorm(swin_x)
        swin_x = self.swin.pos_drop(swin_x)

        swin_x0, h0, w0, swin_y0 = self.swin.layers[0](swin_x, h, w)
        B, L0, C0 = swin_y0.shape
        swin_y0 = self.drop(swin_y0)

        swin_x1, h1, w1, swin_y1 = self.swin.layers[1](swin_x0, h0, w0)
        B, L1, C1 = swin_y1.shape
        swin_y1 = self.drop(swin_y1)

        swin_x2, h2, w2, swin_y2 = self.swin.layers[2](swin_x1, h1, w1)
        B, L2, C2 = swin_y2.shape
        swin_y2 = self.drop(swin_y2)

        # print("最后一个阶段swin_x3与swin_y3相同")
        swin_x3, h3, w3, swin_y3 = self.swin.layers[3](swin_x2, h2, w2)
        swin_y3 = self.swin.norm(swin_y3)
        B, L3, C3 = swin_y3.shape
        swin_y3 = self.drop(swin_y3)

        swin_0_0  = swin_y0.view(B, 56, 56, 96).permute(0, 3, 1, 2)  # 2  96 64 64       bs, 64, 64, 64
        swin_x1_1 = swin_y1.view(B, 28, 28, 192).permute(0, 3, 1, 2)  # 2  192 32 32   128 32 32===>
        swin_x2_2 = swin_y2.view(B, 14, 14, 384).permute(0, 3, 1, 2)  # 2  384 16 16   256 16 16==>

        swin_x3_3 = swin_y3.view(B, 7, 7, 768).permute(0, 3, 1, 2)  # 2  768 8 8     bs  512 8 8

        # 1.把每个staget的特征加起来
        x3=swin_x3_3+self.conv3(x3)
        # x3=self.resblock3(0)
        # x3=swin_x3_3*self.conv9(x3_t)+self.conv3(x3)*self.conv9(x3_t)

        x2=swin_x2_2+self.conv2(x2)
        # x2= swin_x2_2 * self.conv8(x2_t) +self.conv2(x2) * self.conv8(x2_t)

        x1=swin_x1_1+self.conv1(x1)
        # x1 =swin_x1_1 * self.conv7(x1_t) + self.conv1(x1) * self.conv7(x1_t)

        x0=swin_0_0+self.conv0(x0)
        # x0 = swin_0_0 * self.conv6(x0_t) + self.conv0(x0) * self.conv6(x0_t)

        #  经过空间注意力
        x3_att = self.spation(x3)*x3
        x2_att = self.spation(x2)*x2
        x1_att = self.spation(x1)*x1
        x0_att = self.spation(x0)*x0

        #经过通道注意力

        x3_3_channel = self.se3(x3)
        x2_2_channel = self.se2(x2)
        x1_1_channel = self.se1(x1)
        x0_0_channel = self.se0(x0)

        x3=x3_att+x3_3_channel
        x2=x2_att+x2_2_channel
        x1=x1_att+x1_1_channel
        x0=x0_att+x0_0_channel

        x3=self.resblock3(x3)
        x2=self.resblock2(x2)
        x1=self.resblock1(x1)
        x0=self.resblock0(x0)

        # x3=self.conv9(x3)
        # x2=self.conv8(x2)
        # x1=self.conv7(x1)
        # x0=self.conv6(x0)

        d3 = self.decoder3(x3)
        x2=torch.cat([x2,d3],dim=1)
        x2=self.cro2(x2)
        x2_out=x2


        d2 = self.decoder2(x2)

        x1=torch.cat([x1,d2],dim=1)
        x1=self.cro1(x1)
        x1_out=x1


        d1 = self.decoder1(x1)
        x0=torch.cat([x0,d1],dim=1)
        x0=self.cro0(x0)

        # 在编码的每个阶段提取边缘信息
        up2_double = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


        final2x = self.final2(x2_out)
        final1x = self.final1(x1_out)
        final = self.final(x0)

        return up4( final), up4(up2_double( final1x )),up4(up4(final2x)),
