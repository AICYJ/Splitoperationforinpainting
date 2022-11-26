from turtle import forward
import torch.nn as nn
import torch


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def _norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'in':
        layer = nn.InstanceNorm2d(nc, affine=False)
    elif norm_type=='gn':
        layer = nn.GroupNorm(1,nc)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def _activation(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class conv_block(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 padding=0, norm='in', activation='relu', pad_type='zero'):
        super(conv_block, self).__init__()
        if pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        if norm == 'in':
            self.norm = nn.InstanceNorm2d(out_nc, affine=False)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(out_nc, affine=True)
        elif norm=='gn':
            self.norm = nn.GroupNorm(1,out_nc)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported norm type: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2,inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size, stride, 0, dilation, groups, bias)  # padding=0

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class upconv_block(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, bias=True,
                 padding=0, pad_type='zero', norm='none', activation='relu'):
        super(upconv_block, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_nc, out_nc, 4, 2, 1)
        self.act = _activation('relu')
        self.norm = _norm(norm, out_nc)

        self.conv = conv_block(out_nc, out_nc, kernel_size, stride, bias=bias, padding=padding, pad_type=pad_type,
                               norm=norm, activation=activation)

    def forward(self, x):
        x = self.act(self.norm(self.deconv(x)))
        x = self.conv(x)
        return x

class ResBlock_new(nn.Module):
    def __init__(self, nc):
        super(ResBlock_new, self).__init__()
        self.c1 = conv_layer(nc, nc // 4, 3, 1)
        self.d1 = conv_layer(nc // 4, nc // 4, 3, 1, 1)  # rate = 1
        self.d2 = conv_layer(nc // 4, nc // 4, 3, 1, 2)  # rate = 2
        self.d3 = conv_layer(nc // 4, nc // 4, 3, 1, 4)  # rate = 4
        self.d4 = conv_layer(nc // 4, nc // 4, 3, 1, 8)  # rate = 8
        self.act = _activation('relu')
        self.norm = _norm('in', nc)
        self.c2 = conv_layer(nc, nc, 3, 1)  # fusion

    def forward(self, x):
        output1 = self.act(self.norm(self.c1(x)))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        combine = torch.cat([d1, add1, add2, add3], 1)
        output2 = self.c2(self.act(self.norm(combine)))
        output = x + self.norm(output2)
        return output


class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias = True):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU(inplace=False)
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)


    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x
class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class compute_g_spa(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = True):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g
class Splitting(nn.Module):
    #([8, 3, 224, 224])
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :,::2, ::2]

    def odd(self, x):
        return x[:, :,1::2, 1::2]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))

class Splitting2(nn.Module):
    #([8, 3, 224, 224])
    def __init__(self):
        super(Splitting2, self).__init__()

    def even_even(self, x):
        return x[:, :,::2, ::2]

    def even_odd(self, x):
        return x[:, :,::2, 1::2]

    def odd_odd(self, x):
        return x[:, :,1::2, 1::2]
    def odd_even(self, x):
        return x[:, :,1::2,  ::2]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even_even(x), self.odd_odd(x),self.even_odd(x),self.odd_even(x))
# network_G:
#   which_model_G: 'DMFN'
#   in_nc: 4
#   out_nc: 3
#   nf: 64
#   n_res: 8

class ITBConv(nn.Module):
    def __init__(self,in_ch, out_ch,f_size,pd_size,norm='gn'):
        super(ITBConv, self).__init__()
        self.cnn=nn.Conv2d(in_ch,out_ch,f_size,padding=(pd_size,pd_size))
        if norm=='gn':
            self.norm = nn.GroupNorm(1,out_ch)
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(out_ch, affine=False)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x=self.norm(self.relu(self.cnn(x)))
        return x

class ITBLayer(nn.Module):
    def __init__(self,in_ch, out_ch,f_size,pd_size,norm,layer_length):
        super(ITBLayer, self).__init__()
        self.cnn_layer_1=ITBConv(in_ch, out_ch,f_size,pd_size,norm)
        layers=[]
        for _ in range(layer_length-1):
            layer=ITBConv(out_ch, out_ch,f_size,pd_size,norm)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x=self.cnn_layer_1(x)
        x=self.layers(x)
        return x

class ITBLayer1(nn.Module):
    def __init__(self,in_ch, out_ch,f_size,pd_size,norm):
        super(ITBLayer1, self).__init__()
        self.cnn_layer_1=ITBConv(in_ch, out_ch,f_size,pd_size,norm)
        
    
    def forward(self, x):
        x=self.cnn_layer_1(x)
        return x

class ITB(nn.Module):
    def __init__(self,in_ch, out_ch,f_size,pd_size,norm,layer_length):
        super(ITB,self).__init__()
        self.cnn0=ITBLayer(in_ch, out_ch,f_size,pd_size,norm,layer_length)
        self.cnn1=nn.Conv2d(out_ch,out_ch,f_size,padding=(pd_size,pd_size))
        self.cnn2=ITBLayer(out_ch, out_ch,f_size,pd_size,norm,layer_length)
        self.cnn3=ITBLayer(out_ch, out_ch,f_size,pd_size,norm,layer_length)
        self.cnn4=ITBLayer(out_ch, out_ch,f_size,pd_size,norm,layer_length)
        self.cnn5=ITBLayer(out_ch, out_ch,f_size,pd_size,norm,layer_length)
        self.cnn6=ITBLayer(out_ch, out_ch,f_size,pd_size,norm,layer_length)
        self.max=nn.MaxPool2d(f_size+2,stride=1,padding=pd_size+1)
        self.min=nn.MaxPool2d(f_size+2,stride=1,padding=pd_size+1)
        self.avg=nn.AvgPool2d(f_size+2,stride=1,padding=pd_size+1)
        self.tanh=nn.Tanh()
        self.channelcnn = cnn1x1(out_ch, out_ch, bias=False)

    def forward(self, x):
        x0=self.cnn0(x)
        xp=self.tanh(self.cnn1(x0))
        xma=self.cnn2(self.max(x0))
        xmi=self.cnn3(self.min(-1*x0))
        xav=self.cnn4(self.avg(x0))
        xam=xma+xmi
        xs=xam*xp
        xir=self.cnn5(xs+xav)
        xout=self.channelcnn(self.cnn6(xs+xir+x0))
        return xout

class ITB1(nn.Module):
    def __init__(self,in_ch, out_ch,f_size,pd_size,norm):
        super(ITB1,self).__init__()
        self.cnn0=ITBLayer1(in_ch, out_ch,f_size,pd_size,norm)
        self.cnn1=nn.Conv2d(out_ch,out_ch,f_size,padding=(pd_size,pd_size))
        self.cnn2=ITBLayer1(out_ch, out_ch,f_size,pd_size,norm)
        self.cnn3=ITBLayer1(out_ch, out_ch,f_size,pd_size,norm)
        self.cnn4=ITBLayer1(out_ch, out_ch,f_size,pd_size,norm)
        self.cnn5=ITBLayer1(out_ch, out_ch,f_size,pd_size,norm)
        self.cnn6=ITBLayer1(out_ch, out_ch,f_size,pd_size,norm)
        self.max=nn.MaxPool2d(f_size+2,stride=1,padding=pd_size+1)
        self.min=nn.MaxPool2d(f_size+2,stride=1,padding=pd_size+1)
        self.avg=nn.AvgPool2d(f_size+2,stride=1,padding=pd_size+1)
        self.tanh=nn.Tanh()
        self.channelcnn = cnn1x1(out_ch, out_ch, bias=False)

    def forward(self, x):
        x0=self.cnn0(x)
        xp=self.tanh(self.cnn1(x0))
        xma=self.cnn2(self.max(x0))
        xmi=self.cnn3(self.min(-1*x0))
        xav=self.cnn4(self.avg(x0))
        xam=xma+xmi
        xs=xam*xp
        xir=self.cnn5(xs+xav)
        xout=self.channelcnn(self.cnn6(xs+xir+x0))
        return xout


class Split_block(nn.Module):
    def __init__(self,in_ch,o_ch,f_size,activation):
        super(Split_block,self).__init__()
        self.split_1=Splitting()
        self.block_split_e=conv_block(in_ch, o_ch, f_size, stride=1, padding=1, activation=activation)
        self.block_split_o=conv_block(in_ch, o_ch, f_size, stride=1, padding=1, activation=activation)
        self.conv_e=nn.Conv2d(o_ch,o_ch,f_size,padding=(1,1))
        self.conv_o=nn.Conv2d(o_ch,o_ch,f_size,padding=(1,1))
        self.norm_e=nn.GroupNorm(1,o_ch)
        self.norm_o=nn.GroupNorm(1,o_ch)
        self.sig_e=nn.Sigmoid()
        self.sig_o=nn.Sigmoid()

    def forward(self, x):
        x_even_i, x_odd_i = self.split_1(x)#[8, 128, 128]
        x_even=self.block_split_e(x_even_i)
        x_odd=self.block_split_o(x_odd_i)
        x_even=x_even_i+x_even*self.sig_e(self.norm_e(self.conv_e(x_odd)))
        x_odd=x_odd_i+x_odd*self.sig_e(self.norm_e(self.conv_e(x_even)))
        x=torch.cat((x_even,x_odd),1)#[16, 128, 128]
        return x

class Split_block2(nn.Module):
    def __init__(self,in_ch,o_ch,f_size,activation):
        super(Split_block2,self).__init__()
        self.split_1=Splitting()
        self.block_split_e=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.block_split_o=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.conv_e=nn.Conv2d(o_ch,o_ch,f_size,padding=(int((f_size-1)/2),int((f_size-1)/2)))
        self.conv_o=nn.Conv2d(o_ch,o_ch,f_size,padding=(int((f_size-1)/2),int((f_size-1)/2)))
        self.norm_e=nn.GroupNorm(1,o_ch)
        self.norm_o=nn.GroupNorm(1,o_ch)
        self.sig_e=nn.Sigmoid()
        self.sig_o=nn.Sigmoid()

    def forward(self, x):
        x_even_i, x_odd_i = self.split_1(x)#[8, 128, 128]
        x_even=self.block_split_e(x_even_i)
        x_odd=self.block_split_o(x_odd_i)
        x_even_l=x_even+x_even*self.sig_e(self.norm_e(self.conv_e(x_odd)))
        x_odd_l=x_odd+x_odd*self.sig_e(self.norm_e(self.conv_e(x_even)))
        x=torch.cat((x_even_l,x_odd_l),1)#[16, 128, 128]
        return x

class Split_block3(nn.Module):
    def __init__(self,in_ch,o_ch,f_size,activation):
        super(Split_block3,self).__init__()
        self.split_1=Splitting()
        self.block_split_e=conv_block(in_ch, o_ch, f_size, stride=1, padding=1, activation=activation)
        self.block_split_o=conv_block(in_ch, o_ch, f_size, stride=1, padding=1, activation=activation)
        self.conv_e=nn.Conv2d(o_ch,o_ch,f_size,padding=(1,1))
        self.conv_o=nn.Conv2d(o_ch,o_ch,f_size,padding=(1,1))
        self.norm_e=nn.GroupNorm(1,o_ch)
        self.norm_o=nn.GroupNorm(1,o_ch)
        self.sig_e=nn.Sigmoid()
        self.sig_o=nn.Sigmoid()

    def forward(self, x):
        x_even_i, x_odd_i = self.split_1(x)#[8, 128, 128]
        x_even=self.block_split_e(x_even_i)
        x_odd=self.block_split_o(x_odd_i)
        x_even_l=x_even+x_odd*self.sig_e(self.norm_e(self.conv_e(x_odd)))
        x_odd_l=x_odd+x_even*self.sig_e(self.norm_e(self.conv_e(x_even)))
        x=torch.cat((x_even_l,x_odd_l),1)#[16, 128, 128]
        return x


class Split_block4(nn.Module):
    def __init__(self,in_ch,o_ch,f_size,activation):
        super(Split_block4,self).__init__()
        self.split_1=Splitting()
        self.block_split_e=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.block_split_o=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.conv_e=nn.Conv2d(o_ch,o_ch,1,padding=(0,0))
        self.conv_o=nn.Conv2d(o_ch,o_ch,1,padding=(0,0))
        self.norm_e=nn.GroupNorm(1,o_ch)
        self.norm_o=nn.GroupNorm(1,o_ch)
        self.sig_e=nn.Sigmoid()
        self.sig_o=nn.Sigmoid()

    def forward(self, x):
        x_even_i, x_odd_i = self.split_1(x)#[8, 128, 128]
        x_even=self.block_split_e(x_even_i)
        x_odd=self.block_split_o(x_odd_i)
        x_even_l=x_even+x_even*self.sig_e(self.norm_e(self.conv_e(x_odd)))
        x_odd_l=x_odd+x_odd*self.sig_e(self.norm_e(self.conv_e(x_even)))
        x=torch.cat((x_even_l,x_odd_l),1)#[16, 128, 128]
        return x

class Split_block5(nn.Module):
    def __init__(self,in_ch,o_ch,f_size,activation):
        super(Split_block5,self).__init__()
        self.split_1=Splitting2()
        self.block_split_e=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.block_split_o=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.conv_e=nn.Conv2d(o_ch,o_ch,1,padding=(0,0))
        self.conv_o=nn.Conv2d(o_ch,o_ch,1,padding=(0,0))
        self.norm_e=nn.GroupNorm(1,o_ch)
        self.norm_o=nn.GroupNorm(1,o_ch)
        self.sig_e=nn.Sigmoid()
        self.sig_o=nn.Sigmoid()

    def forward(self, x):
        x_even_even, x_odd_odd,x_even_odd,x_odd_even = self.split_1(x)#[8, 128, 128]
        x_even=self.block_split_e(x_even_even)
        x_odd=self.block_split_o(x_odd_odd)
        x_even_l=x_even+x_even_odd*self.sig_e(self.norm_e(self.conv_e(x_odd)))
        x_odd_l=x_odd+x_odd_even*self.sig_e(self.norm_e(self.conv_e(x_even)))
        x=torch.cat((x_even_l,x_odd_l),1)#[16, 128, 128]
        return x

class Split_block6(nn.Module):
    def __init__(self,in_ch,o_ch,f_size,activation):
        super(Split_block6,self).__init__()
        self.split_1=Splitting2()
        self.block_split_e_e=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.block_split_e_o=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.block_split_o_o=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.block_split_o_e=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.conv_e_e=nn.Conv2d(o_ch,o_ch,1,padding=(0,0))
        self.conv_e_o=nn.Conv2d(o_ch,o_ch,1,padding=(0,0))
        self.conv_o_o=nn.Conv2d(o_ch,o_ch,1,padding=(0,0))
        self.conv_o_e=nn.Conv2d(o_ch,o_ch,1,padding=(0,0))
        self.norm_e_e=nn.GroupNorm(1,o_ch)
        self.norm_e_o=nn.GroupNorm(1,o_ch)
        self.norm_o_o=nn.GroupNorm(1,o_ch)
        self.norm_o_e=nn.GroupNorm(1,o_ch)
        self.sig_e_e=nn.Sigmoid()
        self.sig_e_o=nn.Sigmoid()
        self.sig_o_o=nn.Sigmoid()
        self.sig_o_e=nn.Sigmoid()

    def forward(self, x):
        x_even_even, x_odd_odd,x_even_odd,x_odd_even = self.split_1(x)#[8, 128, 128]
        x_e_e=self.block_split_e_e(x_even_even)
        x_e_o=self.block_split_e_o(x_even_odd)
        x_o_o=self.block_split_o_o(x_odd_odd)
        x_o_e=self.block_split_o_e(x_odd_even)
        x_e_e=x_e_e+x_e_o*self.sig_e_e(self.norm_e_e(self.conv_e_e(x_o_e)))
        x_e_o=x_e_o+x_e_e*self.sig_e_o(self.norm_e_o(self.conv_e_o(x_o_o)))
        x_o_o=x_o_o+x_o_e*self.sig_o_o(self.norm_o_o(self.conv_o_o(x_e_o)))
        x_o_e=x_o_e+x_o_o*self.sig_o_e(self.norm_o_e(self.conv_o_e(x_e_e)))
        x=torch.cat((x_e_e,x_e_o,x_o_o,x_o_e),1)#[16, 128, 128]
        return x

class Split_block7(nn.Module):
    def __init__(self,in_ch,o_ch,f_size,activation):
        super(Split_block7,self).__init__()
        self.split_1=Splitting2()
        self.block_split_e_e=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.block_split_e_o=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.block_split_o_o=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.block_split_o_e=conv_block(in_ch, o_ch, f_size, stride=1, padding=int((f_size-1)/2), activation=activation)
        self.conv_e_e=nn.Conv2d(o_ch,o_ch,1,padding=(0,0))
        self.conv_e_o=nn.Conv2d(o_ch,o_ch,1,padding=(0,0))
        self.conv_o_o=nn.Conv2d(o_ch,o_ch,1,padding=(0,0))
        self.conv_o_e=nn.Conv2d(o_ch,o_ch,1,padding=(0,0))
        self.norm_e_e=nn.GroupNorm(1,o_ch)
        self.norm_e_o=nn.GroupNorm(1,o_ch)
        self.norm_o_o=nn.GroupNorm(1,o_ch)
        self.norm_o_e=nn.GroupNorm(1,o_ch)
        self.sig_e_e=nn.Sigmoid()
        self.sig_e_o=nn.Sigmoid()
        self.sig_o_o=nn.Sigmoid()
        self.sig_o_e=nn.Sigmoid()

    def forward(self, x):
        x_even_even, x_odd_odd,x_even_odd,x_odd_even = self.split_1(x)#[8, 128, 128]
        x_e_e=self.block_split_e_e(x_even_even)
        x_e_o=self.block_split_e_o(x_even_odd)
        x_o_o=self.block_split_o_o(x_odd_odd)
        x_o_e=self.block_split_o_e(x_odd_even)
        x_e_e=x_e_e+x_o_o*self.sig_e_e(self.norm_e_e(self.conv_e_e(x_e_e)))
        x_e_o=x_e_o+x_o_e*self.sig_e_o(self.norm_e_o(self.conv_e_o(x_e_o)))
        x_o_o=x_o_o+x_e_e*self.sig_o_o(self.norm_o_o(self.conv_o_o(x_o_o)))
        x_o_e=x_o_e+x_e_o*self.sig_o_e(self.norm_o_e(self.conv_o_e(x_o_e)))
        x=torch.cat((x_e_e,x_e_o,x_o_o,x_o_e),1)#[16, 128, 128]
        return x

class DeSplitting(nn.Module):
    def __init__(self):
        super(DeSplitting, self).__init__()

    def even_even(self, x,split_ch):
        return x[:,0:split_ch,:,:]

    def even_odd(self, x,split_ch):
        return x[:,split_ch:split_ch*2,:,:]

    def odd_odd(self,  x,split_ch):
        return x[:,split_ch*2:split_ch*3,:,:]
    def odd_even(self,  x,split_ch):
        return x[:,split_ch*3:split_ch*4,:,:]

    def forward(self, x):
        split_ch=x.shape[1]//4
        # out_tensor=torch.reshape(x, (x.shape[0],split_ch,x.shape[2]*2,x.shape[3]*2))
        out_tensor=torch.zeros(x.shape[0],split_ch,x.shape[2]*2,x.shape[3]*2)
        out_tensor = out_tensor.type(torch.cuda.FloatTensor)
        # x_cuda = Variable(x, requires_grad=True).cuda()
        out_tensor[:, :,::2, ::2]=self.even_even(x,split_ch)
        out_tensor[:, :,::2, 1::2]=self.even_odd(x,split_ch)
        out_tensor[:, :,1::2, 1::2]=self.odd_odd(x,split_ch)
        out_tensor[:, :,1::2,  ::2]=self.odd_even(x,split_ch)
        return out_tensor