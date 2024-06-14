import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import os
import copy
from utils_clip import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LINEAR_SOFTMAX_ALE(nn.Module):
    def __init__(self, input_dim, attri_dim):
        super(LINEAR_SOFTMAX_ALE, self).__init__()
        self.fc = nn.Linear(input_dim, attri_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, attribute):
        middle = self.fc(x)
        output = self.softmax(middle.mm(attribute))
        return output


class LINEAR_SOFTMAX(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LINEAR_SOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


class LAYER_ALE(nn.Module):
    def __init__(self, input_dim, attri_dim):
        super(LAYER_ALE, self).__init__()
        self.fc = nn.Linear(input_dim, attri_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, attribute):
        batch_size = x.size(0)
        x = torch.mean(x, dim=1)
        x = x.view(batch_size, -1)
        middle = self.fc(x)
        output = self.softmax(middle.mm(attribute))
        return output

class HyperNetStructure():
    def __init__(self, input_dim, num_hiddenLayer, hidden_dim, output_dim):
        self.structure = {}
        if num_hiddenLayer == 0:
            self.structure['InGenerator'] = [input_dim,hidden_dim]
            self.structure['OutGenerator'] = [input_dim,output_dim]
        else:
            self.structure['InGenerator'] = [input_dim,hidden_dim]
            for i in range(num_hiddenLayer):
                self.structure['HiddenGenerator_{}'.format(i+1)] = [input_dim,hidden_dim]
            self.structure['OutGenerator'] = [input_dim,output_dim]
    def get_structure(self):
        return self.structure

class HyperNet(nn.Module):
    def __init__(self, struct):
        super(HyperNet, self).__init__()
        self.struct = struct
        for key,value in struct.items():
            setattr(self,key,nn.Sequential(nn.Linear(value[0],value[1])
                                           ))
    def forward(self,cotrol_signal):
        weights = {}
        for key,_ in self.struct.items():
            weight = getattr(self,key)(cotrol_signal)
            weights[key] = weight
        return weights

class AttrAdapter(nn.Module):
    def __init__(self, input_dim, hypernet_struct, relu=True):
        super().__init__()
        self.hypernet_struct = hypernet_struct
        self.hyper_net = HyperNet(struct=hypernet_struct)
        self.num_generator = len(hypernet_struct)
        self.relu = relu
        i = 0
        incoming = input_dim
        for _,value in hypernet_struct.items():
            setattr(self,'AdapterLayer_{}'.format(i),nn.Linear(incoming,value[1]))
            if i <self.num_generator - 1:
                setattr(self,'AdapterLayer_{}_extend'.format(i),nn.Sequential(
                    nn.LayerNorm(value[1]),
                    nn.ReLU(True),
                    nn.Dropout()
                ))
            incoming = value[1]
            i += 1
        if relu:
            setattr(self,'AdapterLastReLU',nn.ReLU(True))

    def forward(self,control_signal,attr_emb):
        batch_size = control_signal.shape[0]
        num_attr = attr_emb.shape[1]
        attr_emb = attr_emb.t().unsqueeze(dim=0).repeat(batch_size,1,1)
        weights = self.hyper_net(control_signal)
        i = 0
        for key,_ in self.hypernet_struct.items():
            attr_emb =getattr(self,'AdapterLayer_{}'.format(i))(attr_emb)
            weights_extend = weights[key].unsqueeze(dim=1).repeat(1,num_attr,1)
            attr_emb *= weights_extend
            if i < self.num_generator - 1:
                attr_emb = getattr(self,'AdapterLayer_{}_extend'.format(i))(attr_emb)
            i += 1
        if self.relu:
            attr_emb = getattr(self,'AdapterLastReLU')(attr_emb)
        return attr_emb

class CustomModel(nn.Module):
    def __init__(self, clip_model):
        super(CustomModel, self).__init__()
        self.base_model = clip_model
        self.additional_layers = nn.Sequential(
            nn.Linear(512,4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096,2048),
            nn.ReLU(inplace=True)
        )

    def forward(self, attri):
        with torch.no_grad():
            features = self.base_model.encode_text(attri)
        output = self.additional_layers(features)
        return output

class SpatiaAttention(nn.Module):
    def __init__(self, attri_num,kernel_size=7):
        super(SpatiaAttention, self).__init__()
        self.conv1 = nn.Conv2d(2,attri_num,kernel_size,padding=kernel_size//2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)
        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Spectrum(nn.Module):
    def __init__(self,attri_num):
        super(Spectrum, self).__init__()
        self.SpatialAttention = SpatiaAttention(attri_num)
        self.con1x1 = nn.Conv2d(in_channels=attri_num, out_channels=attri_num, kernel_size=1).to(device)

    def forward(self,x):
        spectrum_real = torch.fft.fft(x).real
        spectrum_imag = torch.fft.fft(x).imag
        spectrum_real = self.con1x1(spectrum_real)
        spectrum_imag = self.con1x1(spectrum_imag)
        spectrum_real = F.leaky_relu(spectrum_real)
        spectrum_imag = F.leaky_relu(spectrum_imag)
        x_real = self.SpatialAttention(spectrum_real)
        x_imag = self.SpatialAttention(spectrum_imag)
        output_real = torch.matmul(spectrum_real,x_real)
        output_imag = torch.matmul(spectrum_imag,x_imag)
        output_real = torch.fft.ifft(output_real).real
        output_imag = torch.fft.ifft(output_imag).real
        output_data = torch.stack([output_real,output_imag],dim=-1)
        return output_data[...,0]

class resnet_proto_IoU(nn.Module):
    def __init__(self, opt,attribute_name):
        super(resnet_proto_IoU, self).__init__()
        resnet = models.resnet101()
        num_ftrs = resnet.fc.in_features
        if opt.dataset == 'AWA2':
            num_fc = 1000
        elif opt.dataset == 'CUB':
            num_fc = 150
        elif opt.dataset == 'SUN':
            num_fc = 645
        else:
            num_fc = 1000
        resnet.fc = nn.Linear(num_ftrs, num_fc)
        '''modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)'''
        # 01 - load resnet to model1
        if opt.resnet_path != None:
            state_dict = torch.load(opt.resnet_path)
            resnet.load_state_dict(state_dict)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fine_tune(True)

        # 02 - load cls weights
        # we left the entry for several layers, but here we only use layer4：7*7*2048
        self.dim_dict = {'layer1': 56*56, 'layer2': 28*28, 'layer3': 14*14, 'layer4': 7*7, 'avg_pool': 1*1}
        self.channel_dict = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avg_pool': 2048}
        self.kernel_size = {'layer1': 56, 'layer2': 28, 'layer3': 14, 'layer4': 7, 'avg_pool': 1}
        self.extract = ['layer4']  # 'layer1', 'layer2', 'layer3', 'layer4'
        self.epsilon = 1e-4

        self.softmax = nn.Softmax(dim=1)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        self.conv1x1_layer = nn.Conv2d(2048,85,kernel_size=1)
        attri_num = (int)(opt.attri_num)

        self.proto_net = nn.Sequential(
            nn.Linear(attri_num,4096),    # attri_number
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096,2048),
            nn.LeakyReLU(inplace=True),
        )

        emb_dim = attribute_name.shape[1]
        AttrHyperNet_struct = HyperNetStructure(input_dim=emb_dim,num_hiddenLayer=opt.nhiddenlayers,
                                                hidden_dim=emb_dim*2,output_dim=emb_dim)
        self.AttrHyperNet_struct = AttrHyperNet_struct.get_structure()
        self.attr_adapter = AttrAdapter(input_dim=emb_dim, hypernet_struct=self.AttrHyperNet_struct, relu=False)
        self.Spectrum = Spectrum(attri_num)
        if opt.dataset == 'CUB':
            self.prototype_vectors = dict()
            for name in self.extract:
                attri_expanded = attribute_name.unsqueeze(2).unsqueeze(3)
                self.prototype_vectors[name] = nn.Parameter(attri_expanded, requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([312, 2048, 1, 1]), requires_grad=True)
            self.conv_layers = nn.Sequential(
                nn.Conv2d(attri_num, 1024, kernel_size=3, padding=1),  # 102,85  #CUB:1024  SUN,AWA2:512
                nn.LeakyReLU(inplace=True),                     #CUB
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(1024, attri_num, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                nn.AdaptiveMaxPool2d(1),
            )
        elif opt.dataset == 'AWA1':
            exit(1)
            self.ALE = LINEAR_SOFTMAX_ALE(input_dim=self.channel_dict['avg_pool'], attri_dim=85)
        elif opt.dataset == 'AWA2':
            self.prototype_vectors = dict()
            for name in self.extract:
                attri_expanded = attribute_name.unsqueeze(2).unsqueeze(3)
                self.prototype_vectors[name] = nn.Parameter(attri_expanded, requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([85, 2048, 1, 1]), requires_grad=True)
            self.conv_layers = nn.Sequential(
                nn.Conv2d(attri_num, 512, kernel_size=3, padding=1),  # 102,85  #CUB:1024  SUN,AWA2:512
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, attri_num, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                nn.AdaptiveMaxPool2d(1),
            )
        elif opt.dataset == 'SUN':
            self.prototype_vectors = dict()
            for name in self.extract:
                attri_expanded = attribute_name.unsqueeze(2).unsqueeze(3)
                self.prototype_vectors[name] = nn.Parameter(attri_expanded, requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([102, 2048, 1, 1]), requires_grad=True)
            self.conv_layers = nn.Sequential(
                nn.Conv2d(attri_num, 512, kernel_size=3, padding=1),  # 102,85  #CUB:1024  SUN,AWA2:512
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, attri_num, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                nn.AdaptiveMaxPool2d(1),
            )
        self.avg_pool = opt.avg_pool   #false

    def forward(self,opt, x,attribute_class, return_map=False):
        """out: predict class, predict attributes, maps, out_feature"""
        # x.shape： torch.Size([64, 3, 224, 224])
        # attribute :torch.Size([85, 40])/torch.Size([85, 50])/torch.Size([85, 10])
        record_features = {}
        batch_size = x.size(0)
        x = self.resnet[0:5](x)  # layer 1
        record_features['layer1'] = x  # [64, 256, 56, 56]
        x = self.resnet[5](x)  # layer 2
        record_features['layer2'] = x  # [64, 512, 28, 28]
        x = self.resnet[6](x)  # layer 3
        record_features['layer3'] = x  # [64, 1024, 14, 14]
        x = self.resnet[7](x)  # layer 4
        record_features['layer4'] = x  # [64, 2048, 7, 7]

        attention = F.conv2d(input=x,weight=self.ALE_vector)
        attention = self.Spectrum(attention)

        control_x = self.conv_layers(attention)

        control_x = control_x.squeeze(-1).squeeze(-1)

        attribute = self.attr_adapter(control_x, attribute_class)

        attention = torch.sum(attention,dim=1).unsqueeze(1)

        x_fea = torch.matmul(x,attention)   #[64,2048,7,7]
        x_fea = (F.adaptive_max_pool2d(x_fea,1)+self.resnet[8](x_fea))/2   #[64,2048]
        x_fea = torch.flatten(x_fea,1)
        x_fea = F.normalize(x_fea,dim=-1)

        attribute = torch.sum(attribute,dim=0)

        cls_weight = self.proto_net(attribute)
        cls_weight = F.normalize(cls_weight,dim=-1)
        if opt.dataset == 'AWA2':
            fea_logits = self.softmax(x_fea @ cls_weight.t()/0.01)
        else:
                fea_logits = x_fea @ cls_weight.t()/0.01
        del x_fea,x
        x_fea = 0
        torch.cuda.empty_cache()
        return fea_logits,control_x

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def _l2_convolution(self, x, prototype_vector, one):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2  # [64, C, W, H]
        x2_patch_sum = F.conv2d(input=x2, weight=one)

        p2 = prototype_vector ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_vector)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast  [64, 312,  W, H]
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)  # [64, 312,  W, H]
        return distances

