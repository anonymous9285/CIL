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
            
            gt_bboxes=data_batch[0]['gt_bboxes'].data[0]
            gt_labels=data_batch[0]['gt_labels'].data[0]
            gt_masks=data_batch[0]['gt_masks'].data[0]
            # print(gt_labels)
            data_batch1={}
            data_batch2={}
            for key,value in data_batch[0].items():
                if key=='img_meta':
                    data_batch1[key]=[[value.data[0][0]]]
                    data_batch2[key]=[[value.data[0][1]]]
                elif key=='img':
                    data_batch1[key]=[value.data[0][[True,False]]]
                    data_batch2[key]=[value.data[0][[False,True]]]

            data_batch1['img_aug']=[]
            data_batch1['img_meta_aug']=[]
            data_batch2['img_aug']=[]
            data_batch2['img_meta_aug']=[]

            with torch.no_grad():
                result1box,result1label = self.model(pseudo_loss=True, return_loss=False, rescale=True, **data_batch1)
                result2box,result2label = self.model(pseudo_loss=True, return_loss=False, rescale=True, **data_batch2)

            if len(result1box)!=0 and len(result2box)!=0:
                #4点水平框 mask
                result1gt_bboxes,result1gt_masks=result2bboxmask(result1box,data_batch1['img_meta'][0][0]['img_shape'],data_batch1['img_meta'][0][0]['scale_factor'],data_batch1['img_meta'][0][0]['pad_shape'],data_batch1['img_meta'][0][0]['flip'])
                result2gt_bboxes,result2gt_masks=result2bboxmask(result2box,data_batch2['img_meta'][0][0]['img_shape'],data_batch2['img_meta'][0][0]['scale_factor'],data_batch2['img_meta'][0][0]['pad_shape'],data_batch2['img_meta'][0][0]['flip'])

                #伪标签与gt计算iou，筛去nms大的
                if len(result1label)>0 and len(gt_bboxes[0])>0:
                    gt=torch.cat([gt_bboxes[0]]*len(result1label),dim=1).reshape(-1,4)
                    pred=torch.cat([torch.from_numpy(result1gt_bboxes)]*len(gt_bboxes[0]),dim=0)
                    if len(pred)==0:
                        pdb.set_trace()
                    iou=bbox_iou(gt, pred).reshape(len(gt_bboxes[0]),-1).to('cuda')
                    classcompare = torch.mm(torch.nn.functional.one_hot(gt_labels[0]-1, 15).float().to('cuda'),torch.nn.functional.one_hot(result1label-1, 15).float().t())
                    index=torch.where(iou*classcompare>0.7)[1].cpu().tolist()
                    indexx=list(set([x for x in range(len(result1label))])-set(index))
                    result1box = torch.index_select(result1box, 0, torch.tensor(indexx).to('cuda'))
                    result1label = torch.index_select(result1label.reshape((-1,1)), 0, torch.tensor(indexx).to('cuda')).reshape((1,-1))[0]
                    result1gt_bboxes=torch.index_select(torch.from_numpy(result1gt_bboxes), 0, torch.LongTensor(indexx))
                    result1gt_masks=result1gt_masks[indexx,:]
                
                if len(result2label)>0 and len(gt_bboxes[1])>0:
                    gt=torch.cat([gt_bboxes[1]]*len(result2label),dim=1).reshape(-1,4)
                    pred=torch.cat([torch.from_numpy(result2gt_bboxes)]*len(gt_bboxes[1]),dim=0)
                    iou=bbox_iou(gt, pred).reshape(len(gt_bboxes[1]),-1).to('cuda')
                    classcompare = torch.mm(torch.nn.functional.one_hot(gt_labels[1]-1, 15).float().to('cuda'),torch.nn.functional.one_hot(result2label-1, 15).float().t())
                    index=torch.where(iou*classcompare>0.7)[1].cpu().tolist()
                    indexx=list(set([x for x in range(len(result2label))])-set(index))
                    result2box = torch.index_select(result2box, 0, torch.tensor(indexx).to('cuda'))
                    result2label = torch.index_select(result2label.reshape((-1,1)), 0, torch.tensor(indexx).to('cuda')).reshape((1,-1))[0]
                    result2gt_bboxes=torch.index_select(torch.from_numpy(result2gt_bboxes), 0, torch.LongTensor(indexx))
                    result2gt_masks=result2gt_masks[indexx,:]

                #筛选置信度低于阈值的
                index1=torch.where(result1box[:,8]>0.5)[0].to('cpu').numpy()
                index2=torch.where(result2box[:,8]>0.5)[0].to('cpu').numpy()
                
                result1box = torch.index_select(result1box, 0, torch.tensor(index1).to('cuda'))
                result1label = torch.index_select(result1label.reshape((-1,1)), 0, torch.tensor(index1).to('cuda')).reshape((1,-1))[0]
                result1gt_bboxes=torch.index_select(result1gt_bboxes, 0, torch.LongTensor(index1))
                result1gt_masks=result1gt_masks[index1,:]
                
                result2box = torch.index_select(result2box, 0, torch.tensor(index2).to('cuda'))
                result2label = torch.index_select(result2label.reshape((-1,1)), 0, torch.tensor(index2).to('cuda')).reshape((1,-1))[0]
                result2gt_bboxes=torch.index_select(result2gt_bboxes, 0, torch.LongTensor(index2))
                result2gt_masks=result2gt_masks[index2,:] 


            gt_bboxes_aug=data_batch[1]['gt_bboxes'].data[0]
            gt_labels_aug=data_batch[1]['gt_labels'].data[0]
            gt_masks_aug=data_batch[1]['gt_masks'].data[0]
            # print(gt_labels_aug)
            data_batch1_aug={}
            data_batch2_aug={}
            for key,value in data_batch[1].items():
                if key=='img_meta':
                    data_batch1_aug[key]=[[value.data[0][0]]]
                    data_batch2_aug[key]=[[value.data[0][1]]]
                elif key=='img':
                    data_batch1_aug[key]=[value.data[0][[True,False]]]
                    data_batch2_aug[key]=[value.data[0][[False,True]]]

            data_batch1_aug['img_aug']=[]
            data_batch1_aug['img_meta_aug']=[]
            data_batch2_aug['img_aug']=[]
            data_batch2_aug['img_meta_aug']=[]

            with torch.no_grad():
                result1box_aug,result1label_aug = self.model(pseudo_loss=True, return_loss=False, rescale=True, **data_batch1_aug)
                result2box_aug,result2label_aug = self.model(pseudo_loss=True, return_loss=False, rescale=True, **data_batch2_aug)

            if len(result1box_aug)!=0 and len(result2box_aug)!=0:
                #4点水平框 mask
                result1gt_bboxes_aug,result1gt_masks_aug=result2bboxmask(result1box_aug,data_batch1_aug['img_meta'][0][0]['img_shape'],data_batch1_aug['img_meta'][0][0]['scale_factor'],data_batch1_aug['img_meta'][0][0]['pad_shape'],data_batch1_aug['img_meta'][0][0]['flip'])
                result2gt_bboxes_aug,result2gt_masks_aug=result2bboxmask(result2box_aug,data_batch2_aug['img_meta'][0][0]['img_shape'],data_batch2_aug['img_meta'][0][0]['scale_factor'],data_batch2_aug['img_meta'][0][0]['pad_shape'],data_batch2_aug['img_meta'][0][0]['flip'])

                #伪标签与gt计算iou，筛去nms大的
                if len(result1label_aug)>0 and len(gt_bboxes_aug[0])>0:
                    gt_aug=torch.cat([gt_bboxes_aug[0]]*len(result1label_aug),dim=1).reshape(-1,4)
                    pred_aug=torch.cat([torch.from_numpy(result1gt_bboxes_aug)]*len(gt_bboxes_aug[0]),dim=0)
                    if len(pred_aug)==0:
                        pdb.set_trace()
                    iou_aug=bbox_iou(gt_aug, pred_aug).reshape(len(gt_bboxes_aug[0]),-1).to('cuda')
                    classcompare_aug = torch.mm(torch.nn.functional.one_hot(gt_labels_aug[0]-1, 15).float().to('cuda'),torch.nn.functional.one_hot(result1label_aug-1, 15).float().t())
                    index_aug=torch.where(iou_aug*classcompare_aug>0.7)[1].cpu().tolist()
                    indexx_aug=list(set([x for x in range(len(result1label_aug))])-set(index_aug))
                    result1box_aug = torch.index_select(result1box_aug, 0, torch.tensor(indexx_aug).to('cuda'))
                    result1label_aug = torch.index_select(result1label_aug.reshape((-1,1)), 0, torch.tensor(indexx_aug).to('cuda')).reshape((1,-1))[0]
                    result1gt_bboxes_aug=torch.index_select(torch.from_numpy(result1gt_bboxes_aug), 0, torch.LongTensor(indexx_aug))
                    result1gt_masks_aug=result1gt_masks_aug[indexx_aug,:]
                
                if len(result2label_aug)>0 and len(gt_bboxes_aug[1])>0:
                    gt_aug=torch.cat([gt_bboxes_aug[1]]*len(result2label_aug),dim=1).reshape(-1,4)
                    pred_aug=torch.cat([torch.from_numpy(result2gt_bboxes_aug)]*len(gt_bboxes_aug[1]),dim=0)
                    iou_aug=bbox_iou(gt_aug, pred_aug).reshape(len(gt_bboxes_aug[1]),-1).to('cuda')
                    classcompare_aug = torch.mm(torch.nn.functional.one_hot(gt_labels_aug[1]-1, 15).float().to('cuda'),torch.nn.functional.one_hot(result2label_aug-1, 15).float().t())
                    index_aug=torch.where(iou_aug*classcompare_aug>0.7)[1].cpu().tolist()
                    indexx_aug=list(set([x for x in range(len(result2label_aug))])-set(index_aug))
                    result2box_aug = torch.index_select(result2box_aug, 0, torch.tensor(indexx_aug).to('cuda'))
                    result2label_aug = torch.index_select(result2label_aug.reshape((-1,1)), 0, torch.tensor(indexx_aug).to('cuda')).reshape((1,-1))[0]
                    result2gt_bboxes_aug=torch.index_select(torch.from_numpy(result2gt_bboxes_aug), 0, torch.LongTensor(indexx_aug))
                    result2gt_masks_aug=result2gt_masks_aug[indexx_aug,:]

                #筛选置信度低于阈值的
                index1_aug=torch.where(result1box_aug[:,8]>0.5)[0].to('cpu').numpy()
                index2_aug=torch.where(result2box_aug[:,8]>0.5)[0].to('cpu').numpy()
                
                result1box_aug = torch.index_select(result1box_aug, 0, torch.tensor(index1_aug).to('cuda'))
                result1label_aug = torch.index_select(result1label_aug.reshape((-1,1)), 0, torch.tensor(index1_aug).to('cuda')).reshape((1,-1))[0]
                result1gt_bboxes_aug=torch.index_select(result1gt_bboxes_aug, 0, torch.LongTensor(index1_aug))
                result1gt_masks_aug=result1gt_masks_aug[index1_aug,:]
                
                result2box_aug = torch.index_select(result2box_aug, 0, torch.tensor(index2_aug).to('cuda'))
                result2label_aug = torch.index_select(result2label_aug.reshape((-1,1)), 0, torch.tensor(index2_aug).to('cuda')).reshape((1,-1))[0]
                result2gt_bboxes_aug=torch.index_select(result2gt_bboxes_aug, 0, torch.LongTensor(index2_aug))
                result2gt_masks_aug=result2gt_masks_aug[index2_aug,:] 

            if len(result1box_aug)!=0 and len(result2box_aug)!=0:
                # pdb.set_trace()
                if not torch.is_tensor(result1gt_bboxes_aug):
                    result1gt_bboxes_aug=torch.from_numpy(result1gt_bboxes_aug)
                data_batch[0]['gt_bboxes'].data[0][0]=torch.cat((data_batch[0]['gt_bboxes'].data[0][0],result1gt_bboxes_aug))
                if not torch.is_tensor(result2gt_bboxes_aug):
                    result2gt_bboxes_aug=torch.from_numpy(result2gt_bboxes_aug)
                data_batch[0]['gt_bboxes'].data[0][1]=torch.cat((data_batch[0]['gt_bboxes'].data[0][1],result2gt_bboxes_aug))
                data_batch[0]['gt_labels'].data[0][0]=torch.cat((data_batch[0]['gt_labels'].data[0][0],result1label_aug.to('cpu')))
                data_batch[0]['gt_labels'].data[0][1]=torch.cat((data_batch[0]['gt_labels'].data[0][1],result2label_aug.to('cpu')))
                if len(data_batch[0]['gt_masks'].data[0][0])==0:
                    data_batch[0]['gt_masks'].data[0][0]=np.uint8(np.empty((0,1024,1024)))
                if len(data_batch[0]['gt_masks'].data[0][1])==0:
                    data_batch[0]['gt_masks'].data[0][1]=np.uint8(np.empty((0,1024,1024)))
                data_batch[0]['gt_masks'].data[0][0]=np.vstack([data_batch[0]['gt_masks'].data[0][0],result1gt_masks_aug])
                data_batch[0]['gt_masks'].data[0][1]=np.vstack([data_batch[0]['gt_masks'].data[0][1],result2gt_masks_aug])

            if len(result1box)!=0 and len(result2box)!=0:
                if not torch.is_tensor(result1gt_bboxes):
                    result1gt_bboxes=torch.from_numpy(result1gt_bboxes)
                data_batch[1]['gt_bboxes'].data[0][0]=torch.cat((data_batch[1]['gt_bboxes'].data[0][0],result1gt_bboxes))
                if not torch.is_tensor(result2gt_bboxes):
                    result2gt_bboxes=torch.from_numpy(result2gt_bboxes)
                data_batch[1]['gt_bboxes'].data[0][1]=torch.cat((data_batch[1]['gt_bboxes'].data[0][1],result2gt_bboxes))
                data_batch[1]['gt_labels'].data[0][0]=torch.cat((data_batch[1]['gt_labels'].data[0][0],result1label.to('cpu')))
                data_batch[1]['gt_labels'].data[0][1]=torch.cat((data_batch[1]['gt_labels'].data[0][1],result2label.to('cpu')))
                if len(data_batch[1]['gt_masks'].data[0][0])==0:
                    data_batch[1]['gt_masks'].data[0][0]=np.uint8(np.empty((0,1024,1024)))
                if len(data_batch[1]['gt_masks'].data[0][1])==0:
                    data_batch[1]['gt_masks'].data[0][1]=np.uint8(np.empty((0,1024,1024)))
                data_batch[1]['gt_masks'].data[0][0]=np.vstack([data_batch[1]['gt_masks'].data[0][0],result1gt_masks])
                data_batch[1]['gt_masks'].data[0][1]=np.vstack([data_batch[1]['gt_masks'].data[0][1],result2gt_masks])


            data_batch_ext={}
            data_batch_ext['img']=data_batch[0]['img']
            data_batch_ext['img_aug']=data_batch[1]['img']
            data_batch_ext['img_meta']=data_batch[0]['img_meta']
            data_batch_ext['img_meta_aug']=data_batch[1]['img_meta']
            data_batch_ext['gt_bboxes']=data_batch[0]['gt_bboxes']
            data_batch_ext['gt_bboxes_aug']=data_batch[1]['gt_bboxes']
            data_batch_ext['gt_labels']=data_batch[0]['gt_labels']
            data_batch_ext['gt_labels_aug']=data_batch[1]['gt_labels']
            data_batch_ext['gt_bboxes_ignore']=data_batch[0]['gt_bboxes_ignore']
            data_batch_ext['gt_bboxes_ignore_aug']=data_batch[1]['gt_bboxes_ignore']
            data_batch_ext['gt_masks']=data_batch[0]['gt_masks']
            data_batch_ext['gt_masks_aug']=data_batch[1]['gt_masks']
            
            # if len(data_batch[0]['gt_labels'].data[0][0])==0 or len(data_batch[0]['gt_labels'].data[0][1])==0:
            #     continue
                            
            self._inner_iter = i
            self.call_hook('before_train_iter')
            outputs = self.batch_processor(
                self.model, data_batch_ext, train_mode=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            # print(self.lut)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
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
