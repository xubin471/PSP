from torch import nn
from torch.nn import functional as F
import torch
from .encoder import Res50Encoder as Encoder
from .loss import Loss
import numpy as np
import matplotlib.pyplot as plt
from .SPM import SPM as SPM
from .PCE import PCE
from .ca import CA
from utils import *
class FewShot(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(replace_stride_with_dilation=[True, True, False])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = 20
        self.emb_dim = 512
        self.loss = Loss()
        self.SPM = SPM()
        self.PCE = PCE()
        self.CA = CA()


    def forward(self, sup_img,qry_img, sup_msk,qry_msk,mode="train"):
        # ================================================================================
        img_size = sup_msk.shape[-2:]
        fts,tao = self.encoder(torch.cat([sup_img,qry_img],dim=0))
        sup_fts,qry_fts, self.sup_t, self.qry_t = fts[:1],fts[1:],tao[:1], tao[1:]
        # ================================================================================
        pred_loss = torch.zeros(1).to(self.device)
        align_loss = torch.zeros(1).to(self.device)
        coarse_loss = torch.zeros(1).to(self.device)
        bd_shape_loss = torch.zeros(1).to(self.device)
        consistency_loss = torch.zeros(1).to(self.device)
        rec_loss = torch.zeros(1).to(self.device)
        # ================================================================================
        # PCE module
        sup_fts = self.PCE(sup_fts.clone())
        qry_fts = self.PCE(qry_fts.clone())
        # ================================================================================
        sup_fts_bak, qry_fts_bak = sup_fts.clone(), qry_fts.clone()
        # ================================================================================

        # ================================================================================
        # SPM module
        sup_glob_proto , shape_scores, _ = self._extract_shape_prototypes(sup_fts,sup_msk)
        qry_fts = qry_fts * shape_scores[...,None,None]
        # ================================================================================
        qry_pred_softmax = self.single_proto_predict(qry_fts,sup_glob_proto,self.qry_t,img_size) # [1 1 64 64]
        if mode == "train":
            coarse_loss += self.loss.coarse_loss(qry_pred_softmax,qry_msk)
        # ================================================================================
        # HPP module
        qry_glob_proto = self._extract_prototypes(qry_fts, torch.argmax(qry_pred_softmax,dim=1))  # [1 512]
        final_proto = self.CA(sup_glob_proto,qry_glob_proto)
        qry_pred_softmax_final = self.single_proto_predict(qry_fts,final_proto,self.qry_t,img_size) #[1 2 256 256]

        # ================================================================================
        if mode == "train":
            align_loss += self.align_loss(sup_fts_bak,qry_fts_bak,sup_msk,qry_pred_softmax_final,shape_scores)
            pred_loss += self.loss.pred_loss(qry_pred_softmax_final,qry_msk)
            # if torch.argmax(qry_pred_softmax_final,dim=1).sum() > 200:
            #     bd_shape_loss += 0.001 * self.loss.bd_shape_loss(qry_pred_softmax_final[:,1:,:,:],sup_msk.unsqueeze(1))
            #     consistency_loss += self.loss.pair_wise_consistency_loss(qry_pred_softmax_final, qry_msk)
            return qry_pred_softmax_final, pred_loss,align_loss+coarse_loss+consistency_loss, bd_shape_loss, rec_loss
        else:
            # show_img([sup_img, sup_msk, qry_img, torch.argmax(qry_pred_logit, dim=1), qry_msk], save=True)

            # from utils import keep_largest_component_by_cc
            # qry_attn = keep_largest_component_by_cc((qry_pred[0][0]>0.5).int().detach().cpu().numpy())
            # qry_attn = torch.from_numpy(qry_attn)[None,None,...].to(self.device)
            # qry_pred_logit = torch.cat([1-qry_pred,qry_pred*qry_attn],dim=1)
            return qry_pred_softmax_final
        # show_img([sup_img,sup_msk,qry_img, torch.argmax(qry_pred_logit,dim=1),qry_msk],save=True)
        # ================================================================================

    def align_loss(self, sup_fts, qry_fts, sup_msk, qry_pred,shape_scores1):
        """
        Args:
            supp_fts: (1 512 64 64)
            qry_fts : (1 512 64 64)
            sup_msk: (1 256 256)
            qry_pred: (1, 2, 256, 256)
            shape_scores1: (1 512)

        """
        qry_pred_mask = qry_pred.argmax(dim=1, keepdim=True).squeeze(1)  # (1 256 256]
        skip = qry_pred_mask.sum() <= 200

        # Define loss
        align_loss = torch.zeros(1).to(self.device)
        if skip:
            return align_loss

        # ================================================================================
        qry_glob_proto, shape_scores, _ = self._extract_shape_prototypes(qry_fts,qry_pred_mask)
        sup_fts = sup_fts * shape_scores[...,None,None]
        # ================================================================================
        sup_pred_softmax = self.single_proto_predict(sup_fts,qry_glob_proto,self.sup_t,sup_msk.shape[-2:]) #[1 1 256 256]
        # =================================================================================
        sup_glob_proto = self._extract_prototypes(sup_fts, torch.argmax(sup_pred_softmax,dim=1))  # [1 512]
        final_proto = self.CA(qry_glob_proto,sup_glob_proto)
        sup_pred_softmax_final = self.single_proto_predict(sup_fts,final_proto,self.sup_t,sup_msk.shape[-2:])
        # =================================================================================
        align_loss += self.loss.align_loss(sup_pred_softmax_final,sup_msk)
        return align_loss

    def _extract_prototypes(self, sup_fts, sup_msk):
        """
        Args:
            sup_fts: tensor (B C h w)
            sup_msk: tensor (B H W)
        """
        sup_fts_trans = F.interpolate(sup_fts, size=sup_msk.shape[-2:], mode='bilinear')  # (1 feat_dim 256 256)
        glob_proto = torch.sum(sup_fts_trans*sup_msk.unsqueeze(1),dim=(-2,-1)) / (sup_msk.sum(dim=(-2,-1)) + 1e-8)
        return glob_proto.unsqueeze(1) #[B 1 512]

    def _extract_shape_prototypes(self, sup_fts, sup_msk):
        """
        Args:
            sup_fts: tensor (B C h w)
            sup_msk: tensor (B H W)
        """
        B,C,_,_ = sup_fts.shape

        if sup_msk.sum()>200:
            shape_scores, rec_loss_ = self.SPM(sup_fts,sup_msk)
        else:
            shape_scores = torch.ones(B,C).to(sup_fts.device) #[B C]
            rec_loss_ = torch.zeros(1).to(sup_fts.device) # (1,)

        sup_fts = sup_fts * shape_scores[...,None,None]
        sup_fts_trans = F.interpolate(sup_fts, size=sup_msk.shape[-2:], mode='bilinear')  # (1 feat_dim 256 256)
        glob_proto = torch.sum(sup_fts_trans*sup_msk.unsqueeze(1),dim=(-2,-1)) / (sup_msk.sum(dim=(-2,-1)) + 1e-8) #[B C]
        return glob_proto.unsqueeze(1), shape_scores,rec_loss_  #[B 1 512]

    def single_proto_predict(self,fts, protos, thresh, img_size):
        """
        Args:
            fts: (B C h w)
            protos: (B 1 C)
            thresh: tensor()
            img_size: (H W)
        Retures:
            preds: (B 1 h w)
        """
        B, proto_num, C = protos.shape
        assert B == 1

        sim = -F.cosine_similarity(fts, protos[:,0,:][..., None, None], dim=1) * self.scaler #[B h w]
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))
        pred_ = pred.unsqueeze(1)

        pred_ = F.interpolate(pred_, img_size, mode="bilinear", align_corners=False)
        pred_softmax = torch.cat([1-pred_,pred_],dim=1)
        return pred_softmax
