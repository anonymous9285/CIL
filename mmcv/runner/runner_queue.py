import logging
import os.path as osp
from shlex import join
import time
from tkinter import W
from unittest import result

import torch
import torch.nn as nn                           
import torch.nn.functional as F 
import random
import math
import cv2
import copy
import mmcv
from . import hooks
from .checkpoint import load_checkpoint, save_checkpoint
from .hooks import (CheckpointHook, Hook, IterTimerHook, LrUpdaterHook,
                    OptimizerHook, lr_updater)
from .log_buffer import LogBuffer
from .priority import get_priority
from .utils import get_dist_info, get_host_info, get_time_str, obj_from_dict
import pdb
import numpy as np
from .transforms import BboxTransform,MaskTransform
from pycocotools import mask as maskUtils 
# from .. import (ImageTransform, BboxTransform, MaskTransform,
#                          SegMapTransform, Numpy2Tensor)

from .transforms_rbbox import gt_mask_bp_obbs_list,choose_best_Rroi_batch,dbbox2roi
import os
import matplotlib.pyplot as plt
wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
classthreshold=torch.tensor([0,0.6,0.7,0.8,0.8,0.5,0.4,0.4,0.6,0.7,0.6,0.7,0.88,0.6,0.6,0.6])
def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)

    #Intersection area
    # Intersection area 这里没有对inter_area为负的情况进行判断，后面计算出来的IOU就可能是负的
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def result2bboxmask(result,img_shape, scale_factor,pad_shape,flip):

    # result=torch.trunc(result)
    xmin, ymin, xmax, ymax = result[:,:8:2].min(1)[0], result[:,1::2].min(1)[0], result[:,:8:2].max(1)[0], result[:,1::2].max(1)[0]
    # width, height = xmax - xmin, ymax - ymin
    gt_bboxes=torch.t(torch.cat((xmin.reshape(1,-1),ymin.reshape(1,-1),(xmax-1).reshape(1,-1),(ymax-1).reshape(1,-1)))).type(torch.float32)
    gt_bboxes=torch.trunc(gt_bboxes)
    # pdb.set_trace()
    gt_masks = []
    for ann in result[:,0:8]:
        # pdb.set_trace()
        rles = maskUtils.frPyObjects([ann.cpu().tolist()], 1024, 1024)
        rle = maskUtils.merge(rles)
        gt_mask=maskUtils.decode(rle)
        # if len(gt_mask[gt_mask>0])==0:
        #     pdb.set_trace()
        gt_masks.append(gt_mask)
    if len(gt_bboxes) == 0:
        pdb.set_trace()
        return None
    bbox_transform=BboxTransform()
    mask_transform=MaskTransform()
    gt_bboxes = bbox_transform(gt_bboxes.cpu().numpy(), img_shape, scale_factor,flip)
    gt_masks = mask_transform(gt_masks, pad_shape,scale_factor, flip)
    # pdb.set_trace()
    return gt_bboxes,gt_masks
    

# 超参数
NUM_GTSTATE=120
NUM_PSESTATE=100

BATCH_SIZE = 32                                # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 2000                          # 记忆库容量
N_ACTIONS = NUM_PSESTATE               
N_STATES = NUM_GTSTATE*129+NUM_PSESTATE*129   

REWARD_THRES=torch.log(torch.tensor(0.5))+torch.tensor(0.1)
loss_lst=[]

# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self):                                                         
        super(Net, self).__init__()                                           

        self.prefc1 = nn.Linear(1024, 512)
        self.prefc1.weight.data.normal_(0, 0.05)
        self.prefc2 = nn.Linear(512, 128)
        self.prefc2.weight.data.normal_(0, 0.05)
        self.fc1 = nn.Linear(N_STATES, 64*40)                                     
        self.fc1.weight.data.normal_(0, 0.05)                                 
        self.fc2 = nn.Linear(64*40, 32*40)                                     
        self.fc2.weight.data.normal_(0, 0.05)
        self.out = nn.Linear(32*40, N_ACTIONS)                                     
        self.out.weight.data.normal_(0, 0.05)                                    

    def forward(self, x):    
        if x.dim()==2:
            x=x.unsqueeze(0)                           
        x_pre1 = F.relu(self.prefc1(x[:,:,:1024]))
        x_pre2 = F.relu(self.prefc2(x_pre1))
        x = torch.cat((x_pre2,x[:,:,1024].reshape((len(x),-1,1))),dim=2)
        x = x.reshape((len(x),-1))#torch.Size([1, 15480])
        x = F.relu(self.fc1(x))   #torch.Size([1, 2560])
        x = F.relu(self.fc2(x))   #torch.Size([1, 1280])                                           
        actions_value = self.out(x)#torch.Size([1, 80])                       
        return actions_value                                                    


# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self):                                                         # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net().cuda(), Net().cuda()             # 利用Net创建两个神经网络: 评估网络和目标网络
        # self.eval_net.load_state_dict(torch.load("/workspace/dqn_model11.pkl"))
        # self.target_net.load_state_dict(torch.load("/workspace/dqn_model11.pkl"))
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = [{'s':torch.zeros((NUM_GTSTATE+NUM_PSESTATE,1025)),'a':0,'r':0,'s_':torch.zeros((NUM_GTSTATE+NUM_PSESTATE,1025))}]*MEMORY_CAPACITY             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)

    def choose_action(self, x):#torch.Size([120, 1025]) 
        if np.random.uniform() < EPSILON:                                       
            actions_value = self.eval_net.forward(x) #torch.Size([1, 80])     
            action = torch.max(actions_value, 1)[1].data 
            action = action[0].item()                                                              
        else:                                                                  
            action = np.random.randint(0, N_ACTIONS)  
        # print("action",action)                          
        return action                                                         

    def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
        transition={}
        transition['s']=s
        transition['a']=a
        transition['r']=r
        transition['s_']=s_
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
        self.memory[index] = transition                                      # 置入transition
        self.memory_counter += 1                                                # memory_counter自加1

    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1                                            # 学习步数自加1

        # 抽取记忆库中的批数据
        # sample_index = torch.LongTensor(random.sample(range(BATCH_SIZE), MEMORY_CAPACITY))
        # b_memory = self.memory[sample_index]        
        b_memory = random.sample(self.memory, BATCH_SIZE) 

        b_slst = []
        b_alst = []
        b_rlst = []
        b_s_lst = []
        
        for i in range(0,BATCH_SIZE):
            b_slst.append(b_memory[i]['s'])
            b_alst.append(torch.tensor(b_memory[i]['a']))
            b_rlst.append(torch.tensor(b_memory[i]['r']))
            b_s_lst.append(b_memory[i]['s_'])
        # pdb.set_trace()
        b_s = torch.stack(b_slst,dim=0)
        b_a = torch.stack(b_alst,dim=0).to('cuda')
        b_r = torch.stack(b_rlst,dim=0).to('cuda')
        b_s_ = torch.stack(b_s_lst,dim=0)
        # pdb.set_trace()
        q_eval = self.eval_net(b_s).gather(1, b_a.reshape((-1,1)))     
        q_next = self.target_net(b_s_).detach()
        q_target = b_r.reshape((-1,1)) + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # pdb.set_trace()
        loss = self.loss_func(q_eval, q_target)
        loss_lst.append(loss.item())
        print(loss.item())
    
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数



class Runner(object):
    """A training helper for PyTorch.

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
    """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None):
        assert callable(batch_processor)
        self.dqn=DQN()
        
        self.model = model
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        self.batch_processor = batch_processor

        # create work_dir
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0
        self.lut = torch.zeros((15,10,1024))
        self.lut_num = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # self.register_buffer('lut', torch.zeros(15, 0.24))

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(optimizer, torch.optim,
                                      dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer

    def _add_file_handler(self,
                          logger,
                          filename=None,
                          mode='w',
                          level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, log_dir=None, level=logging.INFO):
        """Init the logger.

        Args:
            log_dir(str, optional): Log file directory. If not specified, no
                log file will be used.
            level (int or str): See the built-in python logging module.

        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        if log_dir and self.rank == 0:
            filename = '{}.log'.format(self.timestamp)
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, 'latest.pth')
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # use relative symlink
        mmcv.symlink(filename, linkpath)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch')


        for i, data_batch in enumerate(data_loader):
            gt_bboxes=data_batch['gt_bboxes'].data[0]
            gt_labels=data_batch['gt_labels'].data[0]
            gt_masks=data_batch['gt_masks'].data[0]

            data_batch1={}
            data_batch2={}
            for key,value in data_batch.items():
                if key=='img_meta':
                    data_batch1[key]=[[value.data[0][0]]]
                    data_batch2[key]=[[value.data[0][1]]]
                elif key=='img':
                    data_batch1[key]=[value.data[0][[True,False]]]
                    data_batch2[key]=[value.data[0][[False,True]]]

            # with torch.no_grad():
            #     #assign pseudo_labels
            #     #8点斜框+score label
            #     result1box,result1label = self.model(pseudo_loss=True, return_loss=False, rescale=True, **data_batch1)
            #     result2box,result2label = self.model(pseudo_loss=True, return_loss=False, rescale=True, **data_batch2)

            result1box=[]
            result1label=[]
            fullname = os.path.join('/workspace/ReDet-master3090/datasets/DOTA_1024/trainval1024/pselabelTxt_0.5', data_batch['img_meta'].data[0][0]['name'].strip('.png') + '.txt')
            f = open(fullname, 'r')
            for line in f.readlines():
                splitlines = line.strip().split(' ')
                result1box.append([float(splitlines[0]), float(splitlines[1]),float(splitlines[2]), float(splitlines[3]),float(splitlines[4]), float(splitlines[5]),float(splitlines[6]), float(splitlines[7]), float(splitlines[8])])
                result1label.append(int(splitlines[9]))
            result1box=torch.from_numpy(np.array(result1box)).to('cuda')
            result1label=torch.from_numpy(np.array(result1label)).to('cuda')

            result2box=[]
            result2label=[]
            fullname = os.path.join('/workspace/ReDet-master3090/datasets/DOTA_1024/trainval1024/pselabelTxt_0.5', data_batch['img_meta'].data[0][1]['name'].strip('.png') + '.txt')
            f = open(fullname, 'r')
            for line in f.readlines():
                splitlines = line.strip().split(' ')
                result2box.append([float(splitlines[0]), float(splitlines[1]),float(splitlines[2]), float(splitlines[3]),float(splitlines[4]), float(splitlines[5]),float(splitlines[6]), float(splitlines[7]), float(splitlines[8])])
                result2label.append(int(splitlines[9]))
            result2box=torch.from_numpy(np.array(result2box)).to('cuda')
            result2label=torch.from_numpy(np.array(result2label)).to('cuda')

            if len(result1box)!=0 and len(result2box)!=0:
                #4点水平框 mask
                result1gt_bboxes,result1gt_masks=result2bboxmask(result1box,data_batch1['img_meta'][0][0]['img_shape'],data_batch1['img_meta'][0][0]['scale_factor'],data_batch1['img_meta'][0][0]['pad_shape'],data_batch1['img_meta'][0][0]['flip'])
                result2gt_bboxes,result2gt_masks=result2bboxmask(result2box,data_batch2['img_meta'][0][0]['img_shape'],data_batch2['img_meta'][0][0]['scale_factor'],data_batch2['img_meta'][0][0]['pad_shape'],data_batch2['img_meta'][0][0]['flip'])

                #伪标签与gt计算iou，筛去nms大的
                if len(result1label)>0:
                    gt=torch.cat([gt_bboxes[0]]*len(result1label),dim=1).reshape(-1,4)
                    pred=torch.cat([torch.from_numpy(result1gt_bboxes)]*len(gt_bboxes[0]),dim=0)
                    if len(pred)==0:
                        pdb.set_trace()
                    iou=bbox_iou(gt, pred).reshape(len(gt_bboxes[0]),-1).to('cuda')
                    classcompare = torch.mm(torch.nn.functional.one_hot(gt_labels[0]-1, 15).float().to('cuda'),torch.nn.functional.one_hot(result1label-1, 15).float().t())
                    index=torch.where(iou*classcompare>0.5)[1].cpu().tolist()
                    indexx=list(set([x for x in range(len(result1label))])-set(index))
                    result1box = torch.index_select(result1box, 0, torch.tensor(indexx).to('cuda'))
                    result1label = torch.index_select(result1label.reshape((-1,1)), 0, torch.tensor(indexx).to('cuda')).reshape((1,-1))[0]
                    result1gt_bboxes=torch.index_select(torch.from_numpy(result1gt_bboxes), 0, torch.LongTensor(indexx))
                    result1gt_masks=result1gt_masks[indexx,:]
                
                if len(result2label)>0:
                    gt=torch.cat([gt_bboxes[1]]*len(result2label),dim=1).reshape(-1,4)
                    pred=torch.cat([torch.from_numpy(result2gt_bboxes)]*len(gt_bboxes[1]),dim=0)
                    iou=bbox_iou(gt, pred).reshape(len(gt_bboxes[1]),-1).to('cuda')
                    classcompare = torch.mm(torch.nn.functional.one_hot(gt_labels[1]-1, 15).float().to('cuda'),torch.nn.functional.one_hot(result2label-1, 15).float().t())
                    index=torch.where(iou*classcompare>0.5)[1].cpu().tolist()
                    indexx=list(set([x for x in range(len(result2label))])-set(index))
                    result2box = torch.index_select(result2box, 0, torch.tensor(indexx).to('cuda'))
                    result2label = torch.index_select(result2label.reshape((-1,1)), 0, torch.tensor(indexx).to('cuda')).reshape((1,-1))[0]
                    result2gt_bboxes=torch.index_select(torch.from_numpy(result2gt_bboxes), 0, torch.LongTensor(indexx))
                    result2gt_masks=result2gt_masks[indexx,:]

                #筛选置信度低于阈值的
                index1=torch.where(result1box[:,8]>0.1)[0].to('cpu').numpy()
                index2=torch.where(result2box[:,8]>0.1)[0].to('cpu').numpy()
                
                result1box = torch.index_select(result1box, 0, torch.tensor(index1).to('cuda'))
                result1label = torch.index_select(result1label.reshape((-1,1)), 0, torch.tensor(index1).to('cuda')).reshape((1,-1))[0]
                result1gt_bboxes=torch.index_select(result1gt_bboxes, 0, torch.LongTensor(index1))
                result1gt_masks=result1gt_masks[index1,:]
                
                result2box = torch.index_select(result2box, 0, torch.tensor(index2).to('cuda'))
                result2label = torch.index_select(result2label.reshape((-1,1)), 0, torch.tensor(index2).to('cuda')).reshape((1,-1))[0]
                result2gt_bboxes=torch.index_select(result2gt_bboxes, 0, torch.LongTensor(index2))
                result2gt_masks=result2gt_masks[index2,:]

                # #筛选一定数量的伪标签
                if len(result1label)>int(NUM_PSESTATE/2):#随机sample
                    # N,C = result1box.shape
                    # S = int(NUM_PSESTATE/2)
                    # pdb.set_trace()
                    # index = torch.LongTensor(random.sample(range(N), S))
                    _,index = result1box[:,-1].topk(int(NUM_PSESTATE/2),dim=0, largest=True, sorted=False)
                    result1box = torch.index_select(result1box, 0, index.to('cuda'))
                    result1label = torch.index_select(result1label.reshape((-1,1)), 0, index.to('cuda')).reshape((1,-1))[0]
                    result1gt_bboxes=torch.index_select(result1gt_bboxes, 0, index.to('cpu'))
                    result1gt_masks=result1gt_masks[index.to('cpu'),:]
                # else:#填充
                #     # pdb.set_trace()
                #     result1box=torch.cat((result1box,torch.zeros((50-len(result1label),9)).to('cuda')))
                #     result1label=torch.cat((result1label,torch.zeros(50-len(result1label)).to('cuda')))

                if len(result2label)>int(NUM_PSESTATE/2):#随机sample
                    # N,C = result2box.shape
                    # S = int(NUM_PSESTATE/2)
                    # pdb.set_trace()
                    # index = torch.LongTensor(random.sample(range(N), S))
                    _,index = result2box[:,-1].topk(int(NUM_PSESTATE/2),dim=0, largest=True, sorted=False)
                    result2box = torch.index_select(result2box, 0, index.to('cuda'))
                    result2label = torch.index_select(result2label.reshape((-1,1)), 0, index.to('cuda')).reshape((1,-1))[0]
                    result2gt_bboxes=torch.index_select(result2gt_bboxes, 0, index.to('cpu'))
                    result2gt_masks=result2gt_masks[index.to('cpu'),:]
                # else:#填充
                #     # pdb.set_trace()
                #     result2box=torch.cat((result2box,torch.zeros((50-len(result2label),9)).to('cuda')))
                #     result2label=torch.cat((result2label,torch.zeros(50-len(result2label)).to('cuda')))

               
            #利用强化筛选置信度和特征相似度
            if len(result1label)!=0 and len(result2label)!=0:
                pseudo_bboxes=[result1gt_bboxes,result2gt_bboxes]#x1,y1,x2,y2 (2,50,)
                pseudo_labels=[result1label.to('cpu'),result2label.to('cpu')]
                pseudo_masks=[result1gt_masks,result2gt_masks]

                with torch.no_grad():
                    #4->5 mask->x,y,w,h,theta
                    x=self.model.module.extract_feat(data_batch['img'].data[0].to('cuda'))
                    gt_obbs = gt_mask_bp_obbs_list(gt_masks)#x,y,w,h,theta
                    gt_rrois=dbbox2roi([torch.from_numpy(choose_best_Rroi_batch(gt_obbs[0])).to('cuda').type(torch.float32),torch.from_numpy(choose_best_Rroi_batch(gt_obbs[1])).to('cuda').type(torch.float32)])
                    gt_rrois[:, 3] = gt_rrois[:, 3] * 1.2
                    gt_rrois[:, 4] = gt_rrois[:, 4] * 1.4
                    gt_rbbox_feats=self.model.module.rbbox_roi_extractor(x[:4], gt_rrois)#torch.Size([6, 256, 7, 7])
                    if self.model.module.rbbox_head.num_shared_convs > 0:
                        for conv in self.model.module.rbbox_head.shared_convs:
                            gt_rbbox_feats = conv(gt_rbbox_feats)
                    if self.model.module.rbbox_head.num_shared_fcs > 0:
                        if self.model.module.rbbox_head.with_avg_pool:
                            gt_rbbox_feats = self.model.module.rbbox_head.avg_pool(gt_rbbox_feats)
                        gt_rbbox_feats = gt_rbbox_feats.view(gt_rbbox_feats.size(0), -1)
                        for fc in self.model.module.rbbox_head.shared_fcs:
                            gt_rbbox_feats = self.model.module.rbbox_head.relu(fc(gt_rbbox_feats))#torch.Size([6, 1024])

                    #create queue
                    gt_labels_lst=torch.cat((gt_labels[0],gt_labels[1]))-1
                    for kk in range(len(gt_labels_lst)):
                        clss=gt_labels_lst[kk]
                        indexx=self.lut_num[clss]%10
                        self.lut_num[clss]=self.lut_num[clss]+1
                        self.lut[clss,indexx] = gt_rbbox_feats[kk]


                    num_gt=len(gt_rbbox_feats)
                    gt_rbbox_feats=torch.cat((gt_rbbox_feats,torch.ones((len(gt_rbbox_feats),1)).to('cuda')),dim=1)
                    if NUM_GTSTATE-len(gt_rbbox_feats)>0:
                        gt_rbbox_feats=torch.cat((gt_rbbox_feats,torch.zeros((NUM_GTSTATE-len(gt_rbbox_feats),1025)).to('cuda')))#torch.Size([40, 1025])

                    pseudo_obbs = gt_mask_bp_obbs_list(pseudo_masks)
                    pseudo_rrois=dbbox2roi([torch.from_numpy(choose_best_Rroi_batch(pseudo_obbs[0])).to('cuda').type(torch.float32),torch.from_numpy(choose_best_Rroi_batch(pseudo_obbs[1])).to('cuda').type(torch.float32)])
                    pseudo_rrois[:, 3] = pseudo_rrois[:, 3] * 1.2
                    pseudo_rrois[:, 4] = pseudo_rrois[:, 4] * 1.4
                    pseudo_rbbox_feats=self.model.module.rbbox_roi_extractor(x[:4], pseudo_rrois)#torch.Size([6, 256, 7, 7])
                    if self.model.module.rbbox_head.num_shared_convs > 0:
                        for conv in self.model.module.rbbox_head.shared_convs:
                            pseudo_rbbox_feats = conv(pseudo_rbbox_feats)
                    if self.model.module.rbbox_head.num_shared_fcs > 0:
                        if self.model.module.rbbox_head.with_avg_pool:
                            pseudo_rbbox_feats = self.model.module.rbbox_head.avg_pool(pseudo_rbbox_feats)
                        pseudo_rbbox_feats = pseudo_rbbox_feats.view(pseudo_rbbox_feats.size(0), -1)
                        for fc in self.model.module.rbbox_head.shared_fcs:
                            pseudo_rbbox_feats = self.model.module.rbbox_head.relu(fc(pseudo_rbbox_feats))#torch.Size([6, 1024])
                    pseudo_rbbox_feats=torch.cat((pseudo_rbbox_feats,torch.cat((result1box[:,8],result2box[:,8])).reshape(-1,1).type(torch.float32)),dim=1) 
                    if NUM_PSESTATE-len(pseudo_rbbox_feats)>0:
                        pseudo_rbbox_feats=torch.cat((pseudo_rbbox_feats,torch.zeros((NUM_PSESTATE-len(pseudo_rbbox_feats),1025)).to('cuda'))) #torch.Size([80, 1025])

                index1=[]
                index2=[]
                while torch.nonzero(pseudo_rbbox_feats).numel()!=0 and num_gt < NUM_GTSTATE:                 
                    a  = self.dqn.choose_action(torch.cat((gt_rbbox_feats,pseudo_rbbox_feats),dim=0))#torch.Size([120, 1025])
                    # print("a",a)
                    # pdb.set_trace()
                    s=gt_rbbox_feats.clone() 
                    s_pse=pseudo_rbbox_feats.clone()                        
                    gt_rbbox_feats[num_gt]=pseudo_rbbox_feats[a]
                    pseudo_rbbox_feats[a]=0
                    num_gt=num_gt+1
                    if torch.nonzero(s_pse[a]).numel()==0:
                        r=-1
                    else:
                        if a<len(result1label):
                            slabel=pseudo_labels[0][a]   
                        else:
                            slabel=pseudo_labels[1][a-len(result1label)]

                        # r= 准确度+类中心+熵（detectmodel）     需要一个function来计算 
                        # class_dictionary = torch.load('new_dictionary11.pt')
                        # class_dictionary1 = torch.load('class_dictionary_tensor11_yuanlai.pt')
                        # cosmatrix = torch.mm(s_pse[a,:1024].unsqueeze(0), class_dictionary.to('cuda').T)/torch.norm(s_pse[a,:1024].unsqueeze(0),p=2,dim=1).reshape((-1,1))/torch.norm(class_dictionary.to('cuda'),p=2,dim=1)     
                        # torch.mm(s[aa,:1024].unsqueeze(0), s[:,:1024].T)/torch.norm(s[aa,:1024].unsqueeze(0),p=2,dim=1).reshape((-1,1))/torch.norm(s[:,:1024],p=2,dim=1)
                        # torch.softmax(torch.mm(s_pse[0,:1024].unsqueeze(0), s[:46,:1024].T)/torch.norm(s_pse[0,:1024].to('cuda').unsqueeze(0),p=2,dim=1).reshape((-1,1))/torch.norm(s[:46,:1024],p=2,dim=1)*5.5,dim=1)[0]
                        # torch.softmax(torch.mm(s_pse[0,:1024].unsqueeze(0), class_dictionary.to('cuda').T)/torch.norm(s_pse[0,:1024].to('cuda').unsqueeze(0),p=2,dim=1).reshape((-1,1))/torch.norm(class_dictionary.to('cuda'),p=2,dim=1)*5.5,dim=1)[0]
                        # torch.softmax(torch.mm(s[0,:1024].unsqueeze(0), class_dictionary.to('cuda').T)/torch.norm(s[0,:1024].to('cuda').unsqueeze(0),p=2,dim=1).reshape((-1,1))/torch.norm(class_dictionary.to('cuda'),p=2,dim=1)*5.5,dim=1)[0]
                        # r=torch.log(s_pse[a,1024])+ torch.softmax(cosmatrix,dim=1)[0][slabel-1]
                        # r=s_pse[a,1024]
                        # r = torch.softmax(cosmatrix*10,dim=1)[0][slabel-1]

                        cls_mean=torch.mean(self.lut[slabel-1],dim=0)
                        r=torch.mm(s_pse[a,:1024].unsqueeze(0), cls_mean.unsqueeze(0).to('cuda').T)/torch.norm(s_pse[a,:1024].unsqueeze(0),p=2,dim=1).reshape((-1,1))/torch.norm(cls_mean.unsqueeze(0).to('cuda'),p=2,dim=1)[0]
                        pdb.set_trace()

                        if r>0.5:
                            r=1
                        else:
                            r=-1

                    self.dqn.store_transition(torch.cat((s,s_pse),dim=0), a, r, torch.cat((gt_rbbox_feats,pseudo_rbbox_feats),dim=0))       
                    if self.dqn.memory_counter > MEMORY_CAPACITY:             
                        self.dqn.learn()
                    if r<0:
                        break
                    else:
                        if a<len(result1label):
                            index1.append(a) 
                        else:
                            index2.append(a-len(result1label))
                if self._epoch>2:
                    data_batch['gt_bboxes'].data[0][0]=torch.cat((data_batch['gt_bboxes'].data[0][0],torch.index_select(result1gt_bboxes, 0, torch.LongTensor(index1))))
                    data_batch['gt_bboxes'].data[0][1]=torch.cat((data_batch['gt_bboxes'].data[0][1],torch.index_select(result2gt_bboxes, 0, torch.LongTensor(index2))))
                    data_batch['gt_labels'].data[0][0]=torch.cat((data_batch['gt_labels'].data[0][0],torch.index_select(result1label.to('cpu'), 0, torch.LongTensor(index1))))
                    data_batch['gt_labels'].data[0][1]=torch.cat((data_batch['gt_labels'].data[0][1],torch.index_select(result2label.to('cpu'), 0, torch.LongTensor(index2))))
                    data_batch['gt_masks'].data[0][0]=np.vstack([data_batch['gt_masks'].data[0][0],result1gt_masks[index1,:]])
                    data_batch['gt_masks'].data[0][1]=np.vstack([data_batch['gt_masks'].data[0][1],result2gt_masks[index2,:]])
                        
                            
            self._inner_iter = i
            self.call_hook('before_train_iter')
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            # print(self.lut)
            self.call_hook('after_train_iter')
            # if self._epoch==11:
            #     with torch.no_grad():
            #         x=self.model.module.extract_feat(data_batch['img'].data[0].to('cuda'))
            #         gt_masks=data_batch['gt_masks'].data[0]
            #         gt_obbs = gt_mask_bp_obbs_list(gt_masks)#x,y,w,h,theta
            #         gt_rrois=dbbox2roi([torch.from_numpy(choose_best_Rroi_batch(gt_obbs[0])).to('cuda').type(torch.float32),torch.from_numpy(choose_best_Rroi_batch(gt_obbs[1])).to('cuda').type(torch.float32)])
            #         gt_rrois[:, 3] = gt_rrois[:, 3] * 1.2
            #         gt_rrois[:, 4] = gt_rrois[:, 4] * 1.4
            #         gt_rbbox_feats=self.model.module.rbbox_roi_extractor(x[:4], gt_rrois)#torch.Size([6, 256, 7, 7])
            #         if self.model.module.rbbox_head.num_shared_convs > 0:
            #             for conv in self.model.module.rbbox_head.shared_convs:
            #                 gt_rbbox_feats = conv(gt_rbbox_feats)
            #         if self.model.module.rbbox_head.num_shared_fcs > 0:
            #             if self.model.module.rbbox_head.with_avg_pool:
            #                 gt_rbbox_feats = self.model.module.rbbox_head.avg_pool(gt_rbbox_feats)
            #             gt_rbbox_feats = gt_rbbox_feats.view(gt_rbbox_feats.size(0), -1)
            #             for fc in self.model.module.rbbox_head.shared_fcs:
            #                 gt_rbbox_feats = self.model.module.rbbox_head.relu(fc(gt_rbbox_feats))#torch.Size([6, 1024])
            #         for conv in self.model.module.rbbox_head.cls_convs:
            #             gt_rbbox_feats = conv(gt_rbbox_feats)
            #         if gt_rbbox_feats.dim() > 2:
            #             if self.model.module.rbbox_head.with_avg_pool:
            #                 gt_rbbox_feats = self.model.module.rbbox_head.avg_pool(gt_rbbox_feats)
            #             gt_rbbox_feats = gt_rbbox_feats.view(gt_rbbox_feats.size(0), -1)
            #         for fc in self.model.module.rbbox_head.cls_fcs:
            #             gt_rbbox_feats = self.model.module.rbbox_head.relu(fc(gt_rbbox_feats))
                    
            #         gt_labels=torch.cat((data_batch['gt_labels'].data[0][0],data_batch['gt_labels'].data[0][1]))-1
            #         gt_labels_a=torch.tensor(list(set(gt_labels.tolist())))
            #         gt_rbbox_feats_lst=torch.Tensor()
            #         for gt_a in gt_labels_a:
            #             iindex=torch.where(gt_labels==gt_a)[0]
            #             gt_rbbox_feats_lst=torch.cat((gt_rbbox_feats_lst,torch.mean(gt_rbbox_feats[iindex],dim=0).to('cpu')),dim=0)
            #         self.lut[gt_labels_a,:] = 0.9 * self.lut[gt_labels_a,:] + (1 - 0.9) * gt_rbbox_feats_lst.reshape(-1,1024) 
            self._iter += 1

        self.call_hook('after_train_epoch')
        #保存字典
        # if self._epoch==11:
        #     torch.save(self.lut, 'new_dictionary{}.pt'.format(self._epoch)) # 保存
        #保存dqn模型
        # torch.save(self.dqn.target_net.state_dict(), "dqn_model{}.pkl".format(self._epoch))
        # #画dqn loss图
        # x_lst=[x for x in range(len(loss_lst))]
        # print(loss_lst)
        # plt.plot(x_lst,loss_lst)
        # plt.ylabel('Loss')
        # plt.xlabel('x')
        # plt.savefig("dqn_loss{}.png".format(self._epoch))
        # plt.show()
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def resume(self, checkpoint, resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            # from .hooks import lr_updater
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = obj_from_dict(
                info, hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)
