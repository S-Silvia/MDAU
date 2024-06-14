from tqdm import tqdm
import torch
import numpy as np
import visual_utils
from statistics import mean
from visual_utils import ImageFilelist, compute_per_class_acc, compute_per_class_acc_gzsl, \
    prepare_attri_label, add_glasso, add_dim_glasso
from CAM_utils import calculate_atten_IoU
import torchvision.transforms as transforms
import random
import torch.nn as nn
import torch.nn.functional as F

class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S = 0.0
        self.best_acc_U = 0.0
        self.acc_list = []
        self.epoch_list = []
    def update(self, it, acc):
        self.acc_list += [acc]
        self.epoch_list += [it]
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.epoch_list += [it]
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U = acc_u
            self.best_acc_S = acc_s


class CategoriesSampler():
    # migrated from Liu et.al., which works well for CUB dataset
    def __init__(self, label_for_imgs, n_batch=1000, n_cls=16, n_per=3, ep_per_batch=1):
        self.n_batch = n_batch # batchs for each epoch
        self.n_cls = n_cls # ways
        self.n_per = n_per # shots
        self.ep_per_batch = ep_per_batch # episodes for each batch, defult set 1
        # print('label_for_imgs:', label_for_imgs[:100])
        # print(np.unique(label_for_imgs))
        self.cat = list(np.unique(label_for_imgs))
        # print('self.cat', len(self.cat))
        # print(self.cat)
        self.catlocs = {}

        for c in self.cat:
            self.catlocs[c] = np.argwhere(label_for_imgs == c).reshape(-1)
        # print('self.catlocs[c]:', self.catlocs[0])

    def __len__(self):
        return  self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                selected_classes = np.random.choice(self.cat, self.n_cls, replace=False)

                for c in selected_classes:
                    l = np.random.choice(self.catlocs[c], self.n_per, replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_cls * n_per
            yield batch.view(-1)



def test_zsl(opt, model, testloader, attribute, test_classes):
    layer_name = model.extract[0]
    GT_targets = []
    predicted_labels = []
    predicted_layers = []
    with torch.no_grad():
        for i, (input, target, impath) in enumerate(testloader):
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            output, _, _, pre_class = model(input, attribute)
            _, predicted_label = torch.max(output.data, 1)
            _, predicted_layer = torch.max(pre_class[layer_name].data, 1)
            predicted_labels.extend(predicted_label.cpu().numpy().tolist())
            predicted_layers.extend(predicted_layer.cpu().numpy().tolist())
            GT_targets = GT_targets + target.data.tolist()
    GT_targets = np.asarray(GT_targets)
    acc_all, acc_avg = compute_per_class_acc(visual_utils.map_label(torch.from_numpy(GT_targets), test_classes).numpy(),
                                     np.array(predicted_labels), test_classes.numpy())
    acc_layer_all, acc_layer_avg = compute_per_class_acc(visual_utils.map_label(torch.from_numpy(GT_targets), test_classes).numpy(),
                                             np.array(predicted_layers), test_classes.numpy())
    if opt.all:
        return acc_all * 100
    else:
        return acc_avg * 100


def calibrated_stacking(opt, output, lam=1e-3):
    """
    output: the output predicted score of size batchsize * 200
    lam: the parameter to control the output score of seen classes.
    self.test_seen_label
    self.test_unseen_label
    :return
    """
    output = output.cpu().numpy()
    seen_L = list(set(opt.test_seen_label.numpy()))
    output[:, seen_L] = output[:, seen_L] - lam
    return torch.from_numpy(output)

def search_calibrated_stacking(opt,output_s,output_u,cs):
    """
    output: the output predicted score of size batchsize * 200
    lam: the parameter to control the output score of seen classes.
    self.test_seen_label
    self.test_unseen_label
    :return
    """
    seen_L = list(set(opt.test_seen_label.numpy()))
    #unseen_L = list(set(opt.test_unseen_label.numpy()))
    if opt.dataset == 'SUN':
        output_s[:, seen_L] = output_s[:, seen_L] * cs
        output_u[:, seen_L] = output_u[:, seen_L] * cs
    else:
        output_s[:, seen_L] = output_s[:, seen_L] - cs
        output_u[:, seen_L] = output_u[:, seen_L] - cs
    output_s = torch.from_numpy(output_s)
    output_u = torch.from_numpy(output_u)
    return output_s,output_u


def compute_class_accuracy_total(opt,true_label, predict_label, classes):
    nclass = len(classes)
    acc_per_class = np.zeros((nclass, 1))
    acc = np.sum(true_label == predict_label) / len(true_label)
    #print(true_label == predict_label)
    for i, class_i in enumerate(classes):
        idx = np.where(true_label == class_i)[0]
        #print('idx:',idx)
        #print('true_label[idx]:',true_label[idx])
        #print('predict_label[idx]:',predict_label[idx])
        acc_per_class[i] = (sum(true_label[idx] == predict_label[idx])*1.0 / len(idx))
    if opt.all:
        return acc
    else:
        return np.mean(acc_per_class)

def test_gzsl(opt, model, testloader_seen,testloader_unseen, attribute,seen_classes,unseen_classes):
    #attribute.shape:[85,50]   test_classes.shape:[10]
    layer_name = model.extract[0]   #layer_name=layer4
    with torch.no_grad():
        #input.shape：[64,3,224,224]
        #target.shape:[64]
        for i, (input, target, impath) in enumerate(testloader_seen):
            if opt.cuda:
                input = input.cuda()    #torch.Size([13, 3, 224, 224])
                target = target.cuda()    #torch.Size([13])
            output_s,_= model(opt,input, attribute)    #[64,50]
            if i ==0:
                gt_s = target.cpu().numpy()
                logits_seen = output_s.cpu().numpy()    #(5882, 50)
            else:
                gt_s = np.concatenate((gt_s,target.cpu().numpy()))   #(5882,)
                logits_seen = np.vstack([logits_seen,output_s.cpu().numpy()])  #(5882, 50)
        for i, (input, target, impath) in enumerate(testloader_unseen):
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            output_u,_ = model(opt,input, attribute)  # [64,50]
            if i ==0:
                gt_u = target.cpu().numpy()
                logits_unseen = output_u.cpu().numpy()
            else:
                gt_u = np.concatenate((gt_u,target.cpu().numpy()))
                logits_unseen = np.vstack([logits_unseen,output_u.cpu().numpy()])   #(7913, 50)
        if opt.calibrated_stacking:
            best_hm = 0
            best_seen = 0
            best_unseen = 0
            for cs in np.arange(0.100, 1.000, 0.001):
                output_s = logits_seen.copy()
                output_u = logits_unseen.copy()
                output_s,output_u = search_calibrated_stacking(opt, output_s,output_u,cs)   #对模型的输出output进行校准
                _, predicted_label_s = torch.max(output_s, 1)   #torch.Size([5882])
                _, predicted_label_u = torch.max(output_u, 1)    #torch.Size([7913])

                acc_all_s = compute_class_accuracy_total(opt, gt_s, np.array(predicted_label_s), seen_classes.numpy())
                acc_all_u = compute_class_accuracy_total(opt, gt_u, np.array(predicted_label_u), unseen_classes.numpy())

                H = (2 * acc_all_s * acc_all_u) / (acc_all_u + acc_all_s)
                #print("acc_all_s,acc_all_u,H:",acc_all_s,acc_all_u,H)
                # print(acc_all_s,acc_all_u)
                if H > best_hm:
                    best_hm = H
                    best_seen = acc_all_s
                    best_unseen = acc_all_u

    return best_hm * 100, best_seen * 100, best_unseen * 100

def test_gzsl_tune_CL(opt, model, testloader, attribute, test_classes, CL=0.98):
    layer_name = model.extract[0]
    GT_targets = []
    predicted_labels = []
    predicted_layers = []
    with torch.no_grad():
        for i, (input, target, impath) in \
                enumerate(testloader):
            if opt.cuda:
                input = input.cuda()
                target = target.cuda()
            output, _, _, pre_class = model(input, attribute)
            if CL:
                output = calibrated_stacking(opt, output, CL)

            _, predicted_label = torch.max(output.data, 1)
            _, predicted_layer = torch.max(pre_class[layer_name].data, 1)
            predicted_labels.extend(predicted_label.cpu().numpy().tolist())
            predicted_layers.extend(predicted_layer.cpu().numpy().tolist())
            GT_targets = GT_targets + target.data.tolist()
    GT_targets = np.asarray(GT_targets)
    acc_all, acc_avg = compute_per_class_acc_gzsl(GT_targets,
                                     np.array(predicted_labels), test_classes.numpy())
    acc_layer_all, acc_layer_avg = compute_per_class_acc_gzsl(GT_targets,
                                             np.array(predicted_layers), test_classes.numpy())
    return acc_all, acc_avg, acc_layer_all, acc_layer_avg


def calculate_average_IoU(whole_IoU, IoU_thr=0.5):
    img_num = len(whole_IoU)
    body_parts = whole_IoU[0].keys()
    body_avg_IoU = {}
    for body_part in body_parts:
        body_avg_IoU[body_part] = []
        body_IoU = []
        for im_id in range(img_num):
            if len(whole_IoU[im_id][body_part]) > 0:
                if_one = []
                for item in whole_IoU[im_id][body_part]:
                    if_one.append(1 if item > IoU_thr else 0)
                body_IoU.append(mean(if_one))
        body_avg_IoU[body_part].append(mean(body_IoU))
    num = 0
    sum = 0
    for part in body_avg_IoU:
        if part != 'tail':
            sum += body_avg_IoU[part][0]
            num += 1
    # print(sum/num *100)
    return body_avg_IoU, sum/num *100


def test_with_IoU(opt, model, testloader, attribute, test_classes, vis_groups=None, vis_root=None,
                  required_num=2, save_att=False, sub_group_dic=None, group_dic=None):
    """
    save feature maps to visualize the activation
    :param model: loaded model
    :param testloader:
    :param attribute: test attributes (model input, no effect to activation maps here)
    :param test_classes: test classes (accuracy input, no effect to activation maps here)
    :param vis_groups: the groups to be shown
    :param vis_layer: the layers to be shown
    :param vis_root: save path to activation maps
    :param required_num: the number images shown in each categories
    :return:
    """
    # print('Calculating the IoU of attention maps, saving attention map to:', save_att)
    layer_name = 'layer4'
    GT_targets = []
    predicted_labels = []
    vis_groups = group_dic

    whole_IoU = []
    # print("vis_groups:", vis_groups)
    # required_num = 2  # requirement denotes the number for each categories
    with torch.no_grad():
        count = dict()
        for i, (input, target, impath) in \
                enumerate(testloader):
            save_att_idx = []
            labels = target.data.tolist()
            for idx in range(len(labels)):
                label = labels[idx]
                if label in count:
                    count[label] = count[label] + 1
                else:
                    count[label] = 1
                if count[label] <= required_num:
                    save_att_idx.append(1)
                else:
                    save_att_idx.append(0)

            if opt.cuda:
                input = input.cuda()
                target = target.cuda()

            output, pre_attri, attention, _ = model(input, attribute)
            # pre_attri.shape : 64， 312
            # attention.shape : 64, 312, 7, 7
            maps = {layer_name: attention['layer4'].cpu().numpy()}
            pre_attri = pre_attri['layer4']
            target_groups = [{} for _ in range(output.size(0))]  # calculate the target groups for each image
            # target_groups is a list of size image_num
            # each item is a dict, including the attention index for each subgroup
            for part in vis_groups.keys():
                sub_group = sub_group_dic[part]
                keys = list(sub_group.keys())

                # sub_activate_id is the attention id for each part in each image. The size is img_num * sub_group_num
                sub_activate_id = []
                for k in keys:
                    sub_activate_id.append(torch.argmax(pre_attri[:, sub_group[k]], dim=1, keepdim=True))
                sub_activate_id = torch.cat(sub_activate_id, dim=1).cpu().tolist()  # (batch_size, sub_group_dim)
                for attention_id, argdims in enumerate(sub_activate_id):
                    target_groups[attention_id][part] = [sub_group[keys[i]][argdim] for i, argdim in enumerate(argdims)]

            KP_root = './data/vis/save_KPs/'
            scale = opt.IoU_scale
            batch_IoU = calculate_atten_IoU(input, impath, save_att_idx, maps, [layer_name], target_groups, KP_root,
                                            save_att=save_att, scale=scale, resize_WH=opt.resize_WH,
                                            KNOW_BIRD_BB=opt.KNOW_BIRD_BB)
            # print('batch_IoU:', batch_IoU)
            whole_IoU += batch_IoU
            _, predicted_label = torch.max(output.data, 1)
            predicted_labels.extend(predicted_label.cpu().numpy().tolist())
            GT_targets = GT_targets + target.data.tolist()
            # break

    body_avg_IoU, mean_IoU = calculate_average_IoU(whole_IoU, IoU_thr=opt.IoU_thr)
    GT_targets = np.asarray(GT_targets)
    acc_all, acc_avg = compute_per_class_acc(visual_utils.map_label(torch.from_numpy(GT_targets), test_classes).numpy(),
                                     np.array(predicted_labels), test_classes.numpy())
    return body_avg_IoU, mean_IoU


def set_randomseed(opt):
    # define random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    # improve the efficiency
    # check CUDA
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def get_loader(opt, data):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if opt.transform_complex:
        train_transform = []
        size = 224
        train_transform.extend([
            transforms.Resize(int(size * 8. / 7.)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize
        ])
        train_transform = transforms.Compose(train_transform)
        test_transform = []
        size = 224
        test_transform.extend([
            transforms.Resize(int(size * 8. / 7.)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose(test_transform)
    else:
        if opt.dataset == 'SUN':
            train_transform = transforms.Compose([
                                      transforms.Resize(448),
                                      transforms.CenterCrop(448),
                                      transforms.ToTensor(),
                                      normalize,
                                  ])
            test_transform = transforms.Compose([
                                            transforms.Resize(448),
                                            transforms.CenterCrop(448),
                                            transforms.ToTensor(),
                                            normalize, ])
        elif opt.dataset == 'AWA2':
            train_transform = transforms.Compose([
                                            transforms.Resize(448),
                                            transforms.CenterCrop(448),
                                            transforms.ToTensor(),
                                            normalize,
                                    ])
            test_transform = transforms.Compose([
                                            transforms.Resize(448),
                                            transforms.CenterCrop(448),
                                            transforms.ToTensor(),
                                            normalize, ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                normalize, ])
    # print("train_transform", train_transform)
    # print("test_transform", test_transform)
    dataset_train = ImageFilelist(opt, data_inf=data,
                                  transform=train_transform,
                                  dataset=opt.dataset,
                                  image_type='trainval_loc')
    if opt.train_mode == 'distributed':
        train_label = dataset_train.image_labels
        # print('len(train_label)', len(train_label))
        sampler = CategoriesSampler(
            train_label,
            n_batch=opt.n_batch,
            n_cls=opt.ways,
            n_per=opt.shots
        )
        trainloader = torch.utils.data.DataLoader(dataset=dataset_train, batch_sampler=sampler, num_workers=4, pin_memory=True)
        # exit()
    elif opt.train_mode == 'random':
        trainloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=opt.batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
    # print('dataset_train.__len__():', dataset_train.__len__())
    # exit()
    dataset_test_unseen = ImageFilelist(opt, data_inf=data,
                                        transform=test_transform,
                                        dataset=opt.dataset,
                                        image_type='test_unseen_loc')
    testloader_unseen = torch.utils.data.DataLoader(
        dataset_test_unseen,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    dataset_test_seen = ImageFilelist(opt, data_inf=data,
                                      transform=transforms.Compose([
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize, ]),
                                      dataset=opt.dataset,
                                      image_type='test_seen_loc')
    testloader_seen = torch.utils.data.DataLoader(
        dataset_test_seen,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # dataset for visualization (CenterCrop)
    dataset_visual = ImageFilelist(opt, data_inf=data,
                                   transform=transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize, ]),
                                   dataset=opt.dataset,
                                   image_type=opt.image_type)

    visloader = torch.utils.data.DataLoader(
        dataset_visual,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    return trainloader, testloader_unseen, testloader_seen, visloader


def get_middle_graph(weight_cpt, model):   #根据给定的权重和模型信息生成一个中间图
    middle_graph = None
    if weight_cpt > 0:
        # creat middle_graph to mask the L_CPT:
        kernel_size = model.kernel_size[model.extract[0]]
        raw_graph = torch.zeros((2 * kernel_size - 1, 2 * kernel_size - 1))
        for x in range(- kernel_size + 1, kernel_size):
            for y in range(- kernel_size + 1, kernel_size):
                raw_graph[x + (kernel_size - 1), y + (kernel_size - 1)] = x ** 2 + y ** 2
        middle_graph = torch.zeros((kernel_size ** 2, kernel_size, kernel_size))
        for x in range(kernel_size):
            for y in range(kernel_size):
                middle_graph[x * kernel_size + y, :, :] = \
                    raw_graph[kernel_size - 1 - x: 2 * kernel_size - 1 - x,
                    kernel_size - 1 - y: 2 * kernel_size - 1 - y]
        middle_graph = middle_graph.cuda()
    return middle_graph


def Loss_fn(fea_logits,opt, loss_log, reg_weight, criterion, criterion_regre, model,
            label_a, label_v,
            realtrain, middle_graph, parts, group_dic, sub_group_dic,pre_atrri):
    ####
    cls_loss_fea = F.cross_entropy(fea_logits, label_v)  # 交叉熵损失
    loss_reg = -fea_logits[torch.arange(fea_logits.size(0)).cuda().long(), label_v].mean()  # 正则化损失
    loss = loss_reg * 0.1 + cls_loss_fea

    return loss

