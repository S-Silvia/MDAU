from __future__ import print_function
import os
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import visual_utils
import sys
import itertools
import random
from visual_utils import ImageFilelist, compute_per_class_acc, compute_per_class_acc_gzsl, \
    prepare_attri_label, add_glasso, add_dim_glasso
from model_proto import resnet_proto_IoU,CustomModel
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import json
from main_utils import test_zsl, calibrated_stacking, test_gzsl, \
    calculate_average_IoU, test_with_IoU
from main_utils import set_randomseed, get_loader, get_middle_graph, Loss_fn, Result
from opt import get_opt
#from utils_clip import *
cudnn.benchmark = True


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

opt = get_opt()
# set random seed 进行随机种子的设置，以实现在随机性的操作中获得可重复的结果
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)



def main():
    # load data
    data = visual_utils.DATA_LOADER(opt)
    opt.test_seen_label = data.test_seen_label  # torch:([5882])

    # define test_classes
    if opt.image_type == 'test_unseen_small_loc':
        test_loc = data.test_unseen_small_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_unseen_loc':
        test_loc = data.test_unseen_loc   #array:(7913,)
        test_classes = data.unseenclasses  #torch:([10])
    elif opt.image_type == 'test_seen_loc':
        test_loc = data.test_seen_loc
        test_classes = data.seenclasses
    else:
        try:
            sys.exit(0)
        except:
            print("choose the image_type in ImageFileList")

    # prepare the attribute labels
    class_attribute = data.attribute   #torch.Size([50, 85])
    attribute_zsl = prepare_attri_label(class_attribute, data.unseenclasses).cuda()  #torch.Size([85, 10])
    attribute_seen = prepare_attri_label(class_attribute, data.seenclasses).cuda()  #torch.Size([85, 40])
    attribute_gzsl = torch.transpose(class_attribute, 1, 0).cuda()   # torch.Size([85, 50])

    # Dataloader for train, test, visual
    trainloader, testloader_unseen, testloader_seen, visloader = get_loader(opt, data)

    # define attribute groups

    if opt.dataset == 'CUB':
        parts = ['head', 'belly', 'breast', 'belly', 'wing', 'tail', 'leg', 'others']
        group_dic = json.load(open(os.path.join(opt.root, 'data', "CUB_200_2011", 'attri_groups_8.json')))
        sub_group_dic = json.load(open(os.path.join(opt.root, 'data', "CUB_200_2011", 'attri_groups_8_layer.json')))
        opt.resnet_path = '/media/cs4007/DATA1/syy/MDAU-main/pretrained_models/resnet101_c.pth.tar'
    elif opt.dataset == 'AWA2':
        parts = ['color', 'texture', 'shape', 'body_parts', 'behaviour', 'nutrition', 'activativity', 'habitat',
                 'character']
        group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_9.json')))
         # {'color': [0, 1, 2, 3, 4, 5, 6, 7], 'texture': [8, 9, 10, 11, 12, 13], 'shape': [14, 15, 16, 17],
         #'body_parts': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 44, 45],
         #'behaviour': [46, 47, 48, 49, 50], 'nutrition': [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
         # 'activativity': [34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
         #'habitat': [63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77],
         #'character': [33, 78, 79, 80, 81, 82, 83, 84]}
        sub_group_dic = {}
        if opt.awa_finetune:
            opt.resnet_path = './pretrained_models/resnet101_awa2.pth.tar'
        else:
            opt.resnet_path = '/media/cs4007/DATA1/syy/MDAU-main/pretrained_models/resnet101-5d3b4d8f.pth'
    elif opt.dataset == 'SUN':
        parts = ['functions', 'materials', 'surface_properties', 'spatial_envelope']
        group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_4.json')))
        sub_group_dic = {}
        opt.resnet_path = '/media/cs4007/DATA1/syy/MDAU-main/pretrained_models/resnet101_sun.pth.tar'
    else:
        opt.resnet_path = '/home/cs4007/SYY/try/pretrained_models/resnet101-5d3b4d8f.pth'

# clip提取attribute
    '''
    if opt.use_clip:
        print("CLIP提取属性特征")
        clip_model, preprocess = load_clip('RN101')
        convert_models_to_fp32(clip_model)  # 将模型中参数都转换为32位浮点
        # 提取属性名称 存入list:50
        attri_list = data.attri_name.tolist()
        if opt.dataset == 'CUB':
            attri = clip.tokenize([f"a photo of a {name} bird" for name in attri_list]).to(device)
        elif opt.dataset == 'AWA2':
            attri = clip.tokenize([f"a photo of a {name} animal" for name in attri_list]).to(device)  # 属性转换为tensor:85*77
        else:
            attri = clip.tokenize([f"a photo related to {name}" for name in attri_list]).to(device)
        # classnames_shape = classnames.shape
        custom_model = CustomModel(clip_model).to(device)
        custom_model.eval()
        attribute_fea = custom_model(attri)  # [85,2048] 每个属性的向量表示
        opt.att_size = attribute_fea.shape[1]  # 85*2048
    else:
        attribute_fea = data.attribute
        '''
    
    attribute_fea = data.attribute
    # initialize model
    print('Create Model...')
    model = resnet_proto_IoU(opt,attribute_fea)
    criterion = nn.CrossEntropyLoss()
    criterion_regre = nn.MSELoss()

    # optimzation weight, only ['final'] + model.extract are used. 配置优化权重的字典
    reg_weight = {'final': {'xe': opt.xe, 'attri': opt.attri, 'regular': opt.regular},
                  'layer4': {'l_xe': opt.l_xe, 'attri': opt.l_attri, 'regular': opt.l_regular,
                             'cpt': opt.cpt},  # l denotes layer
                  }
    ##
    reg_lambdas = {}
    for name in ['final'] + model.extract:
        reg_lambdas[name] = reg_weight[name]
    #print('reg_lambdas:', reg_lambdas)
    #{'final': {'xe': 1.0, 'attri': 0.0001, 'regular': 0.0005}, 'layer4': {'l_xe': 1.0, 'attri': 0.01, 'regular': 5e-07, 'cpt': 2e-09}}

    if torch.cuda.is_available():
        model.cuda()
        attribute_zsl = attribute_zsl.cuda()
        attribute_seen = attribute_seen.cuda()
        attribute_gzsl = attribute_gzsl.cuda()

    layer_name = model.extract[0]  # only use one layer currently : layer4
    # compact loss configuration, define middle_graph
    middle_graph = get_middle_graph(reg_weight[layer_name]['cpt'], model)  #torch.Size([49, 7, 7])

    # train and test
    result_zsl = Result()
    result_gzsl = Result()


    if opt.only_evaluate:
        print('Evaluate ...')
        model.load_state_dict(torch.load(opt.resume))
        model.eval()
        # test zsl
        if not opt.gzsl:
            acc_ZSL = test_zsl(opt, model, testloader_unseen, attribute_zsl, data.unseenclasses)
            print('ZSL test accuracy is {:.1f}%'.format(acc_ZSL))
        else:
            # test gzsl
            acc_GZSL,acc_GZSL_seen,acc_GZSL_unseen = test_gzsl(opt, model,testloader_seen, testloader_unseen, attribute_gzsl, data.seenclasses,data.unseenclasses)



            print('GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'.format(acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL))
    else:
        print('Train and test...')
        #print("opt.nepoch:",opt.nepoch)
        #print("opt.batch_size:", opt.batch_size)
        for epoch in range(opt.nepoch):
            # print("training")
            #opt.only_test = False
            model.train()
            if opt.dataset == 'AWA2':
                pretrain_lr = opt.pretrain_lr * (0.8 ** (epoch // 5))
                current_lr = opt.classifier_lr * (0.8 ** (epoch // 10))
            elif opt.dataset == 'CUB':
                pretrain_lr = opt.pretrain_lr * (0.8 ** (epoch // 10))
                current_lr = opt.classifier_lr * (0.8 ** (epoch // 20))
            else:
                pretrain_lr = opt.pretrain_lr * (0.8 ** (epoch // 5))
                current_lr = opt.classifier_lr * (0.8 ** (epoch // 5))
            realtrain = epoch > opt.pretrain_epoch
            if epoch <= opt.pretrain_epoch:  # pretrain ALE for the first several epoches
                for p in model.parameters():
                    p.requires_grad = True
                for p in model.proto_net.parameters():
                    p.requires_grad = True
                for p in model.conv_layers.parameters():
                    p.requires_grad = True
                for p in model.attr_adapter.parameters():
                    p.requires_grad = True
                '''for p in custom_model.additional_layers.parameters():
                    p.requires_grad = True'''
                for p in model.Spectrum.parameters():
                    p.requires_grad = True
                model_params = [param for param in model.proto_net.parameters()]
                #model_params = [param for param in model.prototype_vectors['layer4']]
                optim_params = [{'params':model_params}]
                #optim_params.append({'params':custom_model.additional_layers.parameters()})
                optim_params.append({'params': model.conv_layers.parameters()})
                optim_params.append({'params': model.attr_adapter.parameters()})
                optim_params.append({'params': model.Spectrum.parameters()})
                optim_params.append({'params': model.ALE_vector})
                optimizer = optim.Adam(optim_params,lr=pretrain_lr, betas=(opt.beta1, 0.999))
                print("lr:",pretrain_lr)
                '''optimizer = optim.Adam(params=[model.ALE_vector],lr=current_lr, betas=(opt.beta1, 0.999))
                optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=pretrain_lr, betas=(opt.beta1, 0.999))'''
            else:
                model.fine_tune()
                optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=current_lr, betas=(opt.beta1, 0.999))
                print("lr:",current_lr)
            # loss for print
            loss_log = {'ave_loss': 0, 'l_xe_final': 0, 'l_attri_final': 0, 'l_regular_final': 0,
                        'l_xe_layer': 0, 'l_attri_layer': 0, 'l_regular_layer': 0, 'l_cpt': 0}

            batch = len(trainloader)   #batch=368
            for i, (batch_input, batch_target, impath) in enumerate(trainloader):
                model.zero_grad()   #将模型中所有可训练参数的梯度置零，以便进行下一轮的梯度更新
                # map target labels
                batch_target = visual_utils.map_label(batch_target, data.seenclasses)  #torch.Size([64]),存储的都是表示类别的索引
                input_v = Variable(batch_input)  #torch.Size([64, 3, 224, 224])
                label_v = Variable(batch_target)  #torch.Size([64])  #记录选中图像的属性索引（0-84）
                if opt.cuda:
                    input_v = input_v.cuda()
                    label_v = label_v.cuda()
                fea_logits,pre_atrri = model(opt,input_v, attribute_seen)
                label_a = attribute_seen[:, label_v].t()  #torch.Size([64, 85]) 记录 一个batch中所有图像的属性向量
                loss = Loss_fn(fea_logits,opt, loss_log, reg_weight, criterion, criterion_regre, model,
                                label_a, label_v,
                               realtrain, middle_graph, parts, group_dic, sub_group_dic,pre_atrri)
                loss_log['ave_loss'] += loss.item()   #累加损失
                loss.backward()
                optimizer.step()   #更新模型参数
            # print('\nLoss log: {}'.format({key: loss_log[key] / batch for key in loss_log}))
            print('\n[Epoch %d, Batch %5d] Train loss: %.3f '
                  % (epoch+1, batch, loss_log['ave_loss'] / batch))
            if (i + 1) == batch or (i + 1) % 200 == 0:   #batch=368
                ###### test #######
                # print("testing")
                model.eval()
                #opt.only_test=True
                # test zsl
                if not opt.gzsl:
                    acc_ZSL = test_zsl(opt, model, testloader_unseen, attribute_zsl, data.unseenclasses)
                    if acc_ZSL > result_zsl.best_acc:
                        # save model state
                        model_save_path = os.path.join('./out/{}_ZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                        torch.save(model.state_dict(), model_save_path)
                        print('model saved to:', model_save_path)
                    result_zsl.update(epoch + 1, acc_ZSL)
                    print('\n[Epoch {}] ZSL test accuracy is {:.1f}%, Best_acc [{:.1f}% | Epoch-{}]'.format(epoch + 1,acc_ZSL,result_zsl.best_acc,result_zsl.best_iter))
                else:
                    # test gzsl
                    acc_GZSL_H,acc_GZSL_seen,acc_GZSL_unseen = test_gzsl(opt, model, testloader_seen,testloader_unseen, attribute_gzsl,data.seenclasses, data.unseenclasses)

                    if acc_GZSL_H > result_gzsl.best_acc:
                        # save model state
                        model_save_path = os.path.join('/media/cs4007/DATA1/syy/MDAU-main/model/out/{}_GZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                        torch.save(model.state_dict(), model_save_path)
                        print('model saved to:', model_save_path)
                    result_gzsl.update_gzsl(epoch+1, acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H)

                    print('\n[Epoch {}] GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'
                          '\n           Best_H [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'.
                          format(epoch+1, acc_GZSL_unseen, acc_GZSL_seen, acc_GZSL_H, result_gzsl.best_acc_U, result_gzsl.best_acc_S,
                    result_gzsl.best_acc, result_gzsl.best_iter))

if __name__ == '__main__':
    main()

