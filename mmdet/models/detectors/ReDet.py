#gs
from __future__ import division

import copy
import torch
import torch.nn as nn
from mmdet.core import RotBox2Polys, polygonToRotRectangle_batch
from mmdet.core import (bbox_mapping, merge_aug_proposals, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms, merge_rotate_aug_proposals,
                        merge_rotate_aug_bboxes, multiclass_nms_rbbox)
from mmdet.core import (build_assigner, bbox2roi, dbbox2roi, bbox2result, build_sampler,
                        dbbox2result, merge_aug_masks, roi2droi, mask2poly,
                        get_best_begin_point, polygonToRotRectangle_batch,
                        gt_mask_bp_obbs_list, choose_best_match_batch,
                        choose_best_Rroi_batch, dbbox_rotate_mapping, bbox_rotate_mapping)

from .base_new import BaseDetectorNew
from .test_mixins import RPNTestMixin
from .. import builder
from ..registry import DETECTORS
import pdb
import numpy as np
from .utis import get_knn, normalize_adj
from ..utils import GCN,GAT
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
import torch.distributed as dist

import random
import os
import cv2

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
               'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
               'helicopter']
color_white = (255, 255, 255)
colormap = [
    (54, 67, 244),
    (99, 30, 233),
    (176, 39, 156),
    (183, 58, 103),
    (181, 81, 63),
    (243, 150, 33),
    (212, 188, 0),
    (136, 150, 0),
    (80, 175, 76),
    (74, 195, 139),
    (57, 220, 205),
    (59, 235, 255),
    (0, 152, 255),
    (34, 87, 255),
    (72, 85, 121),
    (139, 125, 96)]

class PPC(nn.Module):
    def __init__(self):
        super(PPC, self).__init__()
        self.ignore_label = -1

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)
        #n,k     n
        return loss_ppc


class PPD(nn.Module):
    def __init__(self):
        super(PPD, self).__init__()
        self.ignore_label = -1

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()
        return loss_ppd

class FSCELoss(nn.Module):
    def __init__(self):
        super(FSCELoss, self).__init__()
        self.ignore_label = -1

    def forward(self, inputs, targets):
        loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_label)
        return loss

@DETECTORS.register_module
class ReDet(BaseDetectorNew, RPNTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 shared_head_rbbox=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 rbbox_roi_extractor=None,
                 rbbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        assert backbone['type'] == 'ReResNet', 'ReDet only supports ReResNet backbone'
        assert neck['type'] == 'ReFPN', 'ReDet only supports ReFPN neck'
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert rbbox_roi_extractor is not None
        assert rbbox_head is not None

        super(ReDet, self).__init__()

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if shared_head_rbbox is not None:
            self.shared_head_rbbox = builder.build_shared_head(shared_head_rbbox)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        if rbbox_head is not None:
            self.rbbox_roi_extractor = builder.build_roi_extractor(
                rbbox_roi_extractor)
            self.rbbox_head = builder.build_head(rbbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.rbbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # self.gcn_rfeas = GCN(12544, 12544)
        
        # self.gat_rfeas = GAT(nfeat=12544, nhid=12544, nclass=12544,dropout=0.1, nheads=2, alpha=0.01)
        # self.gat_rfeas = GAT(12544, 12544, 12544,0.5, 1, 0.01)
        # self.conv_down0 = nn.Conv2d(in_channels=256, out_channels=10, kernel_size=1, bias=False)
        # self.conv_down = nn.Conv2d(in_channels=266, out_channels=256, kernel_size=1, bias=False)
        # self.conv_down0 = nn.Conv2d(in_channels=259, out_channels=256, kernel_size=1, bias=False)
        self.init_weights(pretrained=pretrained)

        self.num_classes=15
        self.num_prototype=10
        in_channels=1024
        # self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_channels),requires_grad=True)
        # self.prototypes = nn.Parameter(torch.load('prototypes_0.1ms_base.pt').data,requires_grad=True)
        self.prototypes = nn.Parameter(torch.load('prototypes_test1.pt').data,requires_grad=True)
        # trunc_normal_(self.prototypes, std=0.02)
        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.gamma=0.999
        self.update_prototype=True
        self.ppc_criterion = PPC()
        self.ppd_criterion = PPD()
        self.seg_criterion = FSCELoss()
        self.loss_ppc_weight=0.01
        self.loss_ppd_weight=0.01

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(ReDet, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_shared_head_rbbox:
            self.shared_head_rbbox.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_rbbox:
            self.rbbox_roi_extractor.init_weights()
            self.rbbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def l2_normalize(self,x):
        return F.normalize(x, p=2, dim=-1)

    def momentum_update(self,old_value, new_value, momentum, debug=False):
        update = momentum * old_value + (1 - momentum) * new_value
        if debug:
            print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
                momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
                torch.norm(update, p=2)))
        return update

    def distributed_sinkhorn(self,out, sinkhorn_iterations=3, epsilon=0.05):
        L = torch.exp(out / epsilon).t() # K x B
        B = L.shape[1]
        K = L.shape[0]

        # make the matrix sums to 1
        sum_L = torch.sum(L)
        L /= sum_L

        for _ in range(sinkhorn_iterations):
            L /= torch.sum(L, dim=1, keepdim=True)
            L /= K

            L /= torch.sum(L, dim=0, keepdim=True)
            L /= B

        L *= B
        L = L.t()

        indexs = torch.argmax(L, dim=1)
        # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
        L = F.gumbel_softmax(L, tau=0.5, hard=True)

        return L, indexs

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]#b,h,w   pred_seg.view(-1) b*h*w
        mask = (gt_seg == pred_seg.view(-1))  #通过dic 分类分对了

        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]#n,m
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue
            
            q, indexs = self.distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = self.momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(self.l2_normalize(protos),requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            # print('here')
            # print(dist.get_world_size())
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target


    def forward_train(self,#在json文件中，segmentation是8个点，bbox是x1,x2,w,h，外接矩形
                      img,#(3, 1024, 1024)
                      img_meta,#'img_meta': DataContainer({'ori_shape': (1024, 1024, 3), 'img_shape': (1024, 1024, 3), 'pad_shape': (1024, 1024, 3), 'scale_factor': 1.0, 'flip': False})
                      gt_bboxes,#(2,[object数，4()])水平x1,x2,w,h  [x1, y1, x1 + w - 1, y1 + h - 1]
                      gt_labels,#(2,[object数])
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        # print(gt_bboxes)
        # print("gt_labels",gt_labels)
        #print(np.shape(gt_labels))
        x = self.extract_feat(img)
        # print("x",x[0])#tensor
        # print(np.shape(x))#(5,)
        # print(np.shape(x[0]))#torch.Size([2, 256, 256, 256])
        #print(img_meta)#[{'ori_shape': (1024, 1024, 3), 'img_shape': (1024, 1024, 3), 'pad_shape': (1024, 1024, 3), 'scale_factor': 1.0, 'flip': True}, {'ori_shape': (1024, 1024, 3), 'img_shape': (1024, 1024, 3), 'pad_shape': (1024, 1024, 3), 'scale_factor': 1.0, 'flip': False}]
        # print(np.shape(gt_masks[0]))#(1, 1024, 1024)
        # print(np.shape(gt_masks[1]))#(41, 1024, 1024)
        #print(np.shape(gt_masks))
        # pdb.set_trace()
        losses = dict()
        # trans gt_masks to gt_obbs
        gt_obbs = gt_mask_bp_obbs_list(gt_masks)#(2,[object数，5()])
        # print(np.shape(gt_obbs))
        # print(gt_obbs)
        #print("RPN forward and loss")
        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)#包含anchor信息
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            # print(np.shape(rpn_outs))
            # print(np.shape(gt_bboxes))
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)

            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        # print(np.shape(proposal_list))#(2,)
        # print("proposal_list",proposal_list[0])
        # print(proposal_list[0].shape)#torch.Size([2000, 5]) [2000, 5(x1,y1, x2,y2, score)]
        # print(proposal_list[1].shape)
        # print(proposal_list[0][0])
        # print(proposal_list[0][1])
        # print(proposal_list[0][2])

        # print("1111")
        # assign gts and sample proposals (hbb assign)
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn[0].assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn[0].sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                # print(np.shape(proposal_list[i]))
                # print(np.shape(gt_bboxes[i]))
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # print(np.shape(rois))#torch.Size([1024, 5])# (batch_ind,x1, y1 , x2, y2)
            # print(rois[0])
            # print(rois[1])
            # print(rois[2])
            bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
            
            rbbox_targets = self.bbox_head.get_target(
                sampling_results, gt_masks, gt_labels, self.train_cfg.rcnn[0])
              
            '''knn1 = get_knn(rois[0:512, ])
            # knn1 = normalize_adj(knn1, type="DAD")
            
            knn2 = get_knn(rois[512:1024, ])
            # knn2 = normalize_adj(knn2, type="DAD")
            
            knn = np.array([np.array(knn1), np.array(knn2)])
            knn = torch.tensor(knn, dtype=torch.float32).cuda()

            
            N, C, H, W = bbox_feats.shape#1024 256 7 7
            
            newfeats = self.gat_rfeas(bbox_feats.reshape([2, 512, 12544]), knn)
            # newfeats = self.gat_rfeas(bbox_featsmi.reshape([2, 512, 490]), knn)
            # print(np.shape(newfeats))
            newfeats = newfeats.reshape(N, C, H, W)
            
            add_feats = torch.add(newfeats, bbox_feats)
            bbox_feats = add_feats'''

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            # print(np.shape(cls_score))#torch.Size([1024, 16])
            # print(np.shape(bbox_pred))#torch.Size([1024, 5])
            #print(cls_score[0])#torch.Size([1024, 16])
            # print(bbox_pred[0])#torch.Size([1024, 5])

            
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *rbbox_targets)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(0, name)] = (value)

        # print("eee")
        # pdb.set_trace()

        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        # print("pos_is_gts",pos_is_gts)判断为pos,是gt的标志
        #print(len(pos_is_gts))
        roi_labels = rbbox_targets[0]
        with torch.no_grad():
            rotated_proposal_list = self.bbox_head.refine_rbboxes(
                roi2droi(rois), roi_labels, bbox_pred, pos_is_gts, img_meta)
        #print(len(rotated_proposal_list))
        # print(np.shape(rotated_proposal_list[0]))#torch.Size([511, 5])
        # print(np.shape(rotated_proposal_list[1]))#torch.Size([471, 5])

        # assign gts and sample proposals (rbb assign)
        if self.with_rbbox:
            bbox_assigner = build_assigner(self.train_cfg.rcnn[1].assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn[1].sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                gt_obbs_best_roi = choose_best_Rroi_batch(gt_obbs[i])
                # print("r2",np.shape(rotated_proposal_list[i]))#旋转的proposal 数量不固定 512-gt?
                # print("r2",np.shape(gt_obbs_best_roi))#旋转的gt与水平的gt一样多
                assign_result = bbox_assigner.assign(
                    rotated_proposal_list[i], gt_obbs_best_roi, gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    rotated_proposal_list[i],
                    torch.from_numpy(gt_obbs_best_roi).float().to(rotated_proposal_list[i].device),
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)


        if self.with_rbbox:
            #区别，rrois就是有角度的
            rrois = dbbox2roi([res.bboxes for res in sampling_results])
            # print(np.shape(rrois))#torch.Size([1023, 6])# (batch_ind, x_ctr, y_ctr, w, h, angle)
            # print("rrois",rrois[0])#tensor
            # print("rrois", rrois[1])
            # print("rrois", rrois[2])
            # print("rrois", rrois[511])
            # print("rrois", rrois[512])
            # print("rrois", rrois[1023])
            # print(np.shape(rrois[0]))#torch.Size([5])
            # feat enlarge
            rrois[:, 3] = rrois[:, 3] * self.rbbox_roi_extractor.w_enlarge
            rrois[:, 4] = rrois[:, 4] * self.rbbox_roi_extractor.h_enlarge
            rbbox_feats = self.rbbox_roi_extractor(x[:self.rbbox_roi_extractor.num_inputs], rrois)
            # print(np.shape(rbbox_feats))#torch.Size([1023, 256, 7, 7])

            if self.with_shared_head_rbbox:
                rbbox_feats = self.shared_head_rbbox(rbbox_feats)
            # print(np.shape(rbbox_feats))
            cls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
            # print(np.shape(cls_score))#torch.Size([1023, 16])
            # print(np.shape(rbbox_pred))#torch.Size([1023, 16*5])

            rbbox_targets = self.rbbox_head.get_target_rbbox(sampling_results, gt_obbs, gt_labels,
                                                             self.train_cfg.rcnn[1])
            # 原始的object标签和位置
            # print("bbox_targets",bbox_targets)
            #print(rbbox_targets[0])#(4,)labels, label_weights, bbox_targets, bbox_weights
            # print()
            #print(np.shape(rbbox_targets[2]))#torch.Size([1024，64])
            #print(rbbox_targets)
            loss_rbbox = self.rbbox_head.loss(cls_score, rbbox_pred, *rbbox_targets)
            for name, value in loss_rbbox.items():
                losses['s{}.{}'.format(1, name)] = (value)
            # print(losses)

        #s2
        # x=self.model.module.extract_feat(data_batch['img'].data[0].to('cuda'))
        # gt_obbs = gt_mask_bp_obbs_list(gt_masks)#x,y,w,h,theta
        gt_rrois=dbbox2roi([torch.from_numpy(choose_best_Rroi_batch(gt_obbs[0])).to('cuda').type(torch.float32),torch.from_numpy(choose_best_Rroi_batch(gt_obbs[1])).to('cuda').type(torch.float32)])
        gt_rrois[:, 3] = gt_rrois[:, 3] * 1.2
        gt_rrois[:, 4] = gt_rrois[:, 4] * 1.4
        gt_rbbox_feats=self.rbbox_roi_extractor(x[:4], gt_rrois)#torch.Size([n, 256, 7, 7])

        if self.rbbox_head.num_shared_convs > 0:
            for conv in self.rbbox_head.shared_convs:
                gt_rbbox_feats = conv(gt_rbbox_feats)
        if self.rbbox_head.num_shared_fcs > 0:
            if self.rbbox_head.with_avg_pool:
                gt_rbbox_feats = self.rbbox_head.avg_pool(gt_rbbox_feats)
            gt_rbbox_feats = gt_rbbox_feats.view(gt_rbbox_feats.size(0), -1)
            for fc in self.rbbox_head.shared_fcs:
                gt_rbbox_feats = self.rbbox_head.relu(fc(gt_rbbox_feats))#torch.Size([6, 1024])
        
        _c = self.feat_norm(gt_rbbox_feats)
        _c = self.l2_normalize(_c)#n,1024

        self.prototypes.data.copy_(self.l2_normalize(self.prototypes))

        # n: h*w, k: num_class, m: num_prototype
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)#n,m,k

        out_seg = torch.amax(masks, dim=1)#n,k
        out_seg = self.mask_norm(out_seg)
        # out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2])
        gt_seg = torch.cat((gt_labels[0]-1,gt_labels[1]-1))

        contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
        # return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}

        #计算第三部分的loss
        loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
        loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)
        loss_ce = self.seg_criterion(out_seg, gt_seg)
        losses['s{}.{}'.format(2, 'loss_ce')] = (0.1* loss_ce.unsqueeze(0))
        losses['s{}.{}'.format(2, 'loss_ppc')] = (self.loss_ppc_weight * loss_ppc.unsqueeze(0))
        losses['s{}.{}'.format(2, 'loss_ppd')] = (self.loss_ppd_weight * loss_ppd.unsqueeze(0))

        # pdb.set_trace()
        # print(self.prototypes.to('cuda:0'))
        # print('save')
        torch.save(self.prototypes, 'prototypesfullbase1.pt')
        # print(self.prototypes.device)
        # print()
        return losses


    def simple_pseudo(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_label = cls_score.argmax(dim=1)
        rrois = self.bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label, bbox_pred, img_meta[0])

        rrois_enlarge = copy.deepcopy(rrois)
        rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor.w_enlarge
        rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor.h_enlarge

        rbbox_feats = self.rbbox_roi_extractor(x[:len(self.rbbox_roi_extractor.featmap_strides)], rrois_enlarge)

        if self.with_shared_head_rbbox:
            rbbox_feats = self.shared_head_rbbox(rbbox_feats)

        rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
        
        det_rbboxes, det_labels = self.rbbox_head.get_det_rbboxes(
            rrois,
            rcls_score,
            rbbox_pred,
            img_meta[0]['img_shape'],
            img_meta[0]['scale_factor'],
            rescale=rescale,
            cfg=self.test_cfg.rcnn)
        # rbbox_results = dbbox2result(det_rbboxes, det_labels,
        #                              self.rbbox_head.num_classes)
        det_labels=det_labels+1
        # pdb.set_trace()
        return det_rbboxes, det_labels



    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_label = cls_score.argmax(dim=1)
        rrois = self.bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label, bbox_pred, img_meta[0])

        rrois_enlarge = copy.deepcopy(rrois)
        rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor.w_enlarge
        rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor.h_enlarge

        rbbox_feats = self.rbbox_roi_extractor(x[:len(self.rbbox_roi_extractor.featmap_strides)], rrois_enlarge)

        if self.with_shared_head_rbbox:
            rbbox_feats = self.shared_head_rbbox(rbbox_feats)

        rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
        det_rbboxes, det_labels = self.rbbox_head.get_det_rbboxes(#rrois修正，并且xywhtheta5->8点
            rrois,
            rcls_score,
            rbbox_pred,
            img_meta[0]['img_shape'],
            img_meta[0]['scale_factor'],
            rescale=rescale,
            cfg=self.test_cfg.rcnn)

        # img_meta[0]['name'].strip('.png')
        # file_handle=open('/workspace/ReDet-master3090/datasets/DOTA_1024/trainval1024/pselabelTxt/{}.txt'.format(img_meta[0]['name'].strip('.png')),mode='w')#w 写入模式
        # for i in range(det_rbboxes.size(0)):
        #     for j in range(det_rbboxes.size(1)):
        #         aa=str(det_rbboxes[i,j].cpu().numpy())
        #         file_handle.write(aa+' ')
        #     bb=str(det_labels[i].cpu().numpy()+1)
        #     file_handle.write(bb)
        #     file_handle.write('\n')
        # file_handle.close()

        # img_meta[0]['name'].strip('.png')
        # file_handle=open('/workspace/ReDet-master3090/demo/data/demotxt5/{}.txt'.format(img_meta[0]['name'].strip('.png')),mode='w')#w 写入模式
        # for i in range(det_rbboxes.size(0)):
        #     for j in range(det_rbboxes.size(1)):
        #         aa=str(det_rbboxes[i,j].cpu().numpy())
        #         file_handle.write(aa+' ')
        #     bb=str(det_labels[i].cpu().numpy()+1)
        #     file_handle.write(bb)
        #     file_handle.write('\n')
        # file_handle.close()

        # imgpath = os.path.join('/workspace/ReDet-master3090/demo/data/images', img_meta[0]['name'])
        # img = cv2.imread(imgpath,cv2.IMREAD_COLOR)
        # for j in range(det_rbboxes.size(0)):
        #     bbox=det_rbboxes[j].cpu().numpy()
        #     color = colormap[int(det_labels[j].cpu().numpy())]
        #     for i in range(3):
        #         cv2.line(img, (int(bbox[i * 2]), int(bbox[i * 2 + 1])), (int(bbox[(i + 1) * 2]), int(bbox[(i + 1) * 2 + 1])), color=color,thickness=2, lineType=cv2.LINE_AA)
        #     cv2.line(img, (int(bbox[6]), int(bbox[7])), (int(bbox[0]), int(bbox[1])), color=color, thickness=2, lineType=cv2.LINE_AA)
  
        #     cv2.putText(img, '%s' % (str(bbox[8])[0:4]), (int(bbox[0]), int(bbox[1]) + 10),color=(0,0,0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75)
       
        # dstpath1 = os.path.join('/workspace/ReDet-master3090/demo/data/result5', img_meta[0]['name'])
        # cv2.imwrite(dstpath1, img)
        # pdb.set_trace()
        rbbox_results = dbbox2result(det_rbboxes, det_labels,self.rbbox_head.num_classes)#将一张图片中的目标按照类别来保存
        
        return rbbox_results

    def aug_test(self, imgs, img_metas, rescale=None):
        proposal_list = self.aug_test_rpn_rotate(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        aug_rbboxes = []
        aug_rscores = []
        for x, img_meta in zip(self.extract_feats(imgs), img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            angle = img_meta[0]['angle']
            if angle == 0:
                proposals = bbox_mapping(proposal_list[0][:, :4], img_shape, scale_factor, flip)
            else:
                proposals = bbox_rotate_mapping(proposal_list[0][:, :4], img_shape, angle)

            rois = bbox2roi([proposals])
            # recompute feature maps to save GPU memory
            roi_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            cls_score, bbox_pred = self.bbox_head(roi_feats)

            bbox_label = cls_score.argmax(dim=1)
            rrois = self.bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label, bbox_pred, img_meta[0])

            rrois_enlarge = copy.deepcopy(rrois)
            rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor.w_enlarge
            rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor.h_enlarge

            rbbox_feats = self.rbbox_roi_extractor(x[:len(self.rbbox_roi_extractor.featmap_strides)], rrois_enlarge)

            if self.with_shared_head_rbbox:
                rbbox_feats = self.shared_head_rbbox(rbbox_feats)

            rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
            rbboxes, rscores = self.rbbox_head.get_det_rbboxes(
                rrois,
                rcls_score,
                rbbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=None)
            aug_rbboxes.append(rbboxes)
            aug_rscores.append(rscores)

        rcnn_test_cfg = self.test_cfg.rcnn
        merged_rbboxes, merged_rscores = merge_rotate_aug_bboxes(
            aug_rbboxes, aug_rscores, img_metas, rcnn_test_cfg
        )
        det_rbboxes, det_rlabels = multiclass_nms_rbbox(
            merged_rbboxes, merged_rscores, rcnn_test_cfg.score_thr,
            rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img)

        if rescale:
            _det_rbboxes = det_rbboxes
        else:
            _det_rbboxes = det_rbboxes.clone()
            _det_rbboxes[:, :4] *= img_metas[0][0]['scale_factor']

        # pdb.set_trace()
        # img_metas[0][0]['name'].strip('.png')
        # file_handle=open('/workspace/ReDet-master3090/datasets/DOTA_1024_ms/trainval1024_ms/pselabelTxt_0.1/{}.txt'.format(img_metas[0][0]['name'].strip('.png')),mode='w')#w 写入模式
        # for i in range(_det_rbboxes.size(0)):
        #     for j in range(_det_rbboxes.size(1)):
        #         aa=str(_det_rbboxes[i,j].cpu().numpy())
        #         file_handle.write(aa+' ')
        #     bb=str(det_rlabels[i].cpu().numpy()+1)
        #     file_handle.write(bb)
        #     file_handle.write('\n')
        # file_handle.close()

        rbbox_results = dbbox2result(_det_rbboxes, det_rlabels, self.rbbox_head.num_classes)
        return rbbox_results
