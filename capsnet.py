import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from layers import ConvCapsules2d, PrimaryCapsules2d
from vb_routing import VariationalBayesRouting2d


class CapsuleNet(nn.Module):
    ''' Example: Simple 3 layer CapsNet '''
    def __init__(self, args):
        super(CapsuleNet, self).__init__()

        self.P = args.pose_dim
        self.PP = int(np.max([2, self.P*self.P]))
        self.A, self.B, self.C, self.D = args.arch[:-1]
        self.n_classes = args.n_classes = args.arch[-1]
        self.in_channels = 3
        self.batch_norm = True

        conv_arch = [8, 16, 32, 64, 128, 256, 256]
        caps_arch = [32, 32, 6]
        deconv_arch = [128, 256]

        self.Conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=conv_arch[0],
            kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.kaiming_uniform_(self.Conv_1.weight)
        if self.batch_norm: self.BN_1 = nn.BatchNorm2d(conv_arch[0])

        self.Conv_2 = nn.Conv2d(in_channels=conv_arch[0], out_channels=conv_arch[1],
            kernel_size=3, stride=2, padding=1, bias=True)
        nn.init.kaiming_uniform_(self.Conv_2.weight)
        if self.batch_norm: self.BN_2 = nn.BatchNorm2d(conv_arch[1])


        self.Conv_3 = nn.Conv2d(in_channels=conv_arch[1], out_channels=conv_arch[2],
            kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.kaiming_uniform_(self.Conv_3.weight)
        if self.batch_norm: self.BN_3 = nn.BatchNorm2d(conv_arch[2])

        self.Conv_4 = nn.Conv2d(in_channels=conv_arch[2], out_channels=conv_arch[3],
            kernel_size=3, stride=2, padding=1, bias=True)
        nn.init.kaiming_uniform_(self.Conv_4.weight)
        if self.batch_norm: self.BN_4 = nn.BatchNorm2d(conv_arch[3])


        self.Conv_5 = nn.Conv2d(in_channels=conv_arch[3], out_channels=conv_arch[4],
            kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.kaiming_uniform_(self.Conv_5.weight)
        if self.batch_norm: self.BN_5 = nn.BatchNorm2d(conv_arch[4])

        self.Conv_6 = nn.Conv2d(in_channels=conv_arch[4], out_channels=conv_arch[4],
            kernel_size=3, stride=2, padding=1, bias=True)
        nn.init.kaiming_uniform_(self.Conv_6.weight)
        if self.batch_norm: self.BN_6 = nn.BatchNorm2d(conv_arch[4])


        self.Conv_7 = nn.Conv2d(in_channels=conv_arch[4], out_channels=conv_arch[5],
            kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.kaiming_uniform_(self.Conv_7.weight)
        if self.batch_norm: self.BN_7 = nn.BatchNorm2d(conv_arch[5])

        self.Conv_8 = nn.Conv2d(in_channels=conv_arch[5], out_channels=conv_arch[5],
            kernel_size=3, stride=2, padding=1, bias=True)
        nn.init.kaiming_uniform_(self.Conv_8.weight)
        if self.batch_norm: self.BN_8 = nn.BatchNorm2d(conv_arch[5])


        self.Conv_9 = nn.Conv2d(in_channels=conv_arch[5], out_channels=conv_arch[6],
            kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.kaiming_uniform_(self.Conv_9.weight)
        if self.batch_norm: self.BN_9 = nn.BatchNorm2d(conv_arch[6])


        self.PrimaryCaps = PrimaryCapsules2d(in_channels=conv_arch[6], out_caps=caps_arch[0],
            kernel_size=9, stride=1, pose_dim=self.P)

        self.SecondaryConvCaps = ConvCapsules2d(in_caps=caps_arch[0], out_caps=caps_arch[1],
            kernel_size=5, stride=2, pose_dim=self.P)#, padding=2)

        self.SecondaryConvRouting = VariationalBayesRouting2d(in_caps=caps_arch[0], out_caps=caps_arch[1],
            kernel_size=5, stride=2, pose_dim=self.P,
            cov='diag', iter=args.routing_iter,
            alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
            Psi0=torch.eye(self.PP), nu0=self.PP+1)


        self.ClassCaps = ConvCapsules2d(in_caps=caps_arch[1], out_caps=caps_arch[2],
            kernel_size=1, stride=1, pose_dim=self.P, share_W_ij=True, coor_add=True)

        self.ClassRouting = VariationalBayesRouting2d(in_caps=caps_arch[1], out_caps=caps_arch[2],
            kernel_size=9, stride=1, pose_dim=self.P, # adjust final kernel_size K depending on input H/W, for H=W=32, K=4.
            cov='diag', iter=args.routing_iter,
            alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
            Psi0=torch.eye(self.PP), nu0=self.PP+1, class_caps=True)

        self.FCN_1 = nn.Linear(in_features=self.PP, out_features=7*7)

        self.ConvTr_1 = nn.ConvTranspose2d(in_channels=1, out_channels=deconv_arch[0], 
            kernel_size=3, stride=1, bias=False)  # , padding=1

        self.Skip_ConvTr1 = nn.ConvTranspose2d(in_channels=caps_arch[1]*self.PP, out_channels=deconv_arch[0], 
            kernel_size=3, stride=1, padding=1, bias=False) 

        self.ConvTr_2 = nn.ConvTranspose2d(in_channels=deconv_arch[1], out_channels=deconv_arch[0], 
            kernel_size=6, stride=2, bias=False) 
        
        self.Skip_ConvTr2 = nn.ConvTranspose2d(in_channels=caps_arch[0]*self.PP, out_channels=deconv_arch[0], 
            kernel_size=3, stride=1, padding=1, bias=False) 

        self.ConvTr_3 = nn.ConvTranspose2d(in_channels=deconv_arch[1], out_channels=deconv_arch[1], 
            kernel_size=9, stride=1, bias=False) 

        self.ConvTr_4 = nn.ConvTranspose2d(in_channels=deconv_arch[1], out_channels=deconv_arch[1], 
            kernel_size=4, stride=2, padding=1, bias=False)

        self.ConvTr_5 = nn.ConvTranspose2d(in_channels=deconv_arch[1], out_channels=deconv_arch[1], 
            kernel_size=4, stride=2, padding=1, bias=False)
        
        self.ConvTr_6 = nn.ConvTranspose2d(in_channels=deconv_arch[1], out_channels=deconv_arch[1], 
            kernel_size=4, stride=2, padding=1, bias=False)
        
        self.ConvTr_7 = nn.ConvTranspose2d(in_channels=deconv_arch[1], out_channels=deconv_arch[1], 
            kernel_size=4, stride=2, padding=1, bias=False)
        
        self.segment_layer = nn.Conv2d(in_channels=deconv_arch[1], out_channels=1,
            kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.kaiming_uniform_(self.segment_layer.weight)


    def forward(self, x):
        print_flag = False
        if print_flag: print('-' * 70)

        # Out ← [?, A, F, F]
        if self.batch_norm: 
          x = F.relu(self.BN_1(self.Conv_1(x)))
          if print_flag: print(f'Conv_1 Shape: {x.shape}')
          x = F.relu(self.BN_2(self.Conv_2(x)))
          if print_flag: print(f'Conv_2 Shape: {x.shape}')
          x = F.relu(self.BN_3(self.Conv_3(x)))
          if print_flag: print(f'Conv_3 Shape: {x.shape}')
          x = F.relu(self.BN_4(self.Conv_4(x)))
          if print_flag: print(f'Conv_4 Shape: {x.shape}')
          x = F.relu(self.BN_5(self.Conv_5(x)))
          if print_flag: print(f'Conv_5 Shape: {x.shape}')
          x = F.relu(self.BN_6(self.Conv_6(x)))
          if print_flag: print(f'Conv_6 Shape: {x.shape}')
          x = F.relu(self.BN_7(self.Conv_7(x)))
          if print_flag: print(f'Conv_7 Shape: {x.shape}')
          x = F.relu(self.BN_8(self.Conv_8(x)))
          if print_flag: print(f'Conv_8 Shape: {x.shape}')
          x = F.relu(self.BN_9(self.Conv_9(x)))
          if print_flag: print(f'Conv_9 Shape: {x.shape}')
        else:
          x = F.relu(self.Conv_1(x))
          if print_flag: print(f'Conv_1 Shape: {x.shape}')
          x = F.relu(self.Conv_2(x))
          if print_flag: print(f'Conv_2 Shape: {x.shape}')
          x = F.relu(self.Conv_3(x))
          if print_flag: print(f'Conv_3 Shape: {x.shape}')
          x = F.relu(self.Conv_4(x))
          if print_flag: print(f'Conv_4 Shape: {x.shape}')
          x = F.relu(self.Conv_5(x))
          if print_flag: print(f'Conv_5 Shape: {x.shape}')
          x = F.relu(self.Conv_6(x))
          if print_flag: print(f'Conv_6 Shape: {x.shape}')
          x = F.relu(self.Conv_7(x))
          if print_flag: print(f'Conv_7 Shape: {x.shape}')
          x = F.relu(self.Conv_8(x))
          if print_flag: print(f'Conv_8 Shape: {x.shape}')
          x = F.relu(self.Conv_9(x))
          if print_flag: print(f'Conv_9 Shape: {x.shape}')

        if print_flag: print(f'Output of conv layers: {x.shape}')
        if print_flag: print('-' * 70)

        # Out ← a [?, B, F, F], v [?, B, P, P, F, F]
        prim_caps = self.PrimaryCaps(x)
        if print_flag: print(f'PrimaryCaps Shape: {prim_caps[0].shape}, {prim_caps[1].shape}')

        
        # Out ← a [?, B, 1, 1, 1, F, F, K, K], v [?, B, C, P*P, 1, F, F, K, K]
        a,v = self.SecondaryConvCaps(*prim_caps)
        if print_flag: print(f'SecondaryConvCaps Shape: {a.shape}, {v.shape}')
        
        # Out ← a [?, C, F, F], v [?, C, P, P, F, F]
        sec_caps = self.SecondaryConvRouting(a,v)
        if print_flag: print(f'SecondaryConvRouting Shape: {sec_caps[0].shape}, {sec_caps[1].shape}')

        # Out ← a [?, D, 1, 1, 1, F, F, K, K], v [?, D, n_classes, P*P, 1, F, F, K, K]
        a,v = self.ClassCaps(sec_caps[0], sec_caps[1])
        if print_flag: print(f'ClassCaps Shape: {a.shape}, {v.shape}')

        # Out ← yhat [?, n_classes], v [?, n_classes, P, P]
        activations, poses = self.ClassRouting(a,v)
        if print_flag: print(f'ClassRouting Shape: {activations.shape}, {poses.shape}')
        
        poses = poses.reshape(*poses.shape[:-2], -1)
        activations = activations.unsqueeze(dim=2).expand(*activations.shape, self.PP)

        poses_prod = activations * poses
        poses_sum = torch.sum(poses_prod, dim=1)
        
        fcn_1 = F.relu(self.FCN_1(poses_sum))
        
        batch_size = poses_sum.shape[0]
        fcn_1 = fcn_1.reshape(batch_size, 1, 7, 7)
        if print_flag: print(f'Dense Shape: {fcn_1.shape}')        
        
        deconv1 = F.relu(self.ConvTr_1(fcn_1))
        if print_flag: print(f'Deconv_1 Shape: {deconv1.shape}')
        
        batch_size, num_caps, P1, P2, F1, F2 = sec_caps[1].shape
        caps_res = sec_caps[1].reshape(batch_size, -1, F1, F2)
        skip1 = self.Skip_ConvTr1(caps_res)
        if print_flag: print(f'Skip1 Shape: {skip1.shape}')

        deconv1 = torch.cat((deconv1, skip1), dim=1)
        if print_flag: print(f'Deconv_1 + Skip1 Shape: {deconv1.shape}')
        
        deconv2 = F.relu(self.ConvTr_2(deconv1))
        if print_flag: print(f'Deconv_2 Shape: {deconv2.shape}')
        
        batch_size, num_caps, P1, P2, F1, F2 = prim_caps[1].shape
        caps_res = prim_caps[1].reshape(batch_size, -1, F1, F2)
        skip2 = self.Skip_ConvTr2(caps_res)
        if print_flag: print(f'Skip2 Shape: {skip2.shape}')

        deconv2 = torch.cat((deconv2, skip2), dim=1)
        if print_flag: print(f'Deconv_2 + Skip2 Shape: {deconv2.shape}')
        
        deconv3 = F.relu(self.ConvTr_3(deconv2))
        if print_flag: print(f'Deconv_3 Shape: {deconv3.shape}')

        deconv4 = F.relu(self.ConvTr_4(deconv3))
        if print_flag: print(f'Deconv_4 Shape: {deconv4.shape}')

        deconv5 = F.relu(self.ConvTr_5(deconv4))
        if print_flag: print(f'Deconv_5 Shape: {deconv5.shape}')

        deconv6 = F.relu(self.ConvTr_6(deconv5))
        if print_flag: print(f'Deconv_6 Shape: {deconv6.shape}')

        deconv7 = F.relu(self.ConvTr_7(deconv6))
        if print_flag: print(f'Deconv_7 Shape: {deconv7.shape}')

        mask_pred = F.sigmoid(self.segment_layer(deconv7))
        if print_flag: print(f'Segment Layer Shape: {mask_pred.shape}')

        '''
        batch_size, n_classes, F1, F2 = yhat.shape
        yhat = yhat.reshape(batch_size, n_classes, 1, F1, F2)
        v = v.reshape(batch_size, n_classes, -1, F1, F2)
        out = torch.cat([yhat, v], dim=2)
        if print_flag: print(f'Out Shape: {out.shape}')
        
        batch_size, n_classes, _ , F1, F2 = out.shape
        out = out.reshape(batch_size, -1, F1, F2)

        out = self.ConvTr_1(out)
        if print_flag: print(f'ConvTr_1 Shape: {out.shape}')
        '''

        return mask_pred