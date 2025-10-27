import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from icecream import ic
import sys

class Masked_MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super(Masked_MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask):
        def masked_softmax(X):
            X_orig = X
            # head*batch, track, feature <- 2
            # new: head, batch, track, feature <- 3
            # head, batch, track, feature
            X = X.reshape(self.num_heads, Q.shape[0], X.shape[1], X.shape[2])
            max_elem = torch.max(X, dim=3)[0]
            mask_1 = mask.reshape(1, mask.shape[0], mask.shape[1], 1)
            mask_2 = mask.reshape(1, mask.shape[0], 1, mask.shape[1])
            if torch.any(torch.isnan(max_elem)):
                print("max_elem NaN")
                sys.exit(1)


            X_temp = (X - max_elem.unsqueeze(3)) * mask_1 * mask_2
            num = torch.exp(X_temp)
            num = num * mask_1 * mask_2
            dem = torch.sum(num, dim=3).unsqueeze(3)
            res = num/(dem + 1e-16)
            res = res.reshape(X_orig.shape[0], X_orig.shape[1], X_orig.shape[2])
            return res

        Q = self.fc_q(Q)*mask.unsqueeze(-1)
        K, V = self.fc_k(K)*mask.unsqueeze(-1), self.fc_v(K)*mask.unsqueeze(-1)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = masked_softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V))
        # Hopefully, for all padded values, A is 0
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O + F.relu(self.fc_o(O)*mask.unsqueeze(-1))
        return O

    def forward_check(self, Q, K, mask):
        def masked_softmax(X):
            X_orig = X
            # head*batch, track, feature <- 2
            # new: head, batch, track, feature <- 3
            # head, batch, track, feature
            X = X.reshape(self.num_heads, Q.shape[0], X.shape[1], X.shape[2])
            max_elem = torch.max(X, dim=3)[0]
            mask_1 = mask.reshape(1, mask.shape[0], mask.shape[1], 1)
            mask_2 = mask.reshape(1, mask.shape[0], 1, mask.shape[1])
            if torch.any(torch.isnan(max_elem)):
                print("max_elem NaN")
                sys.exit(1)


            X_temp = (X - max_elem.unsqueeze(3)) * mask_1 * mask_2
            num = torch.exp(X_temp)
            num = num * mask_1 * mask_2
            dem = torch.sum(num, dim=3).unsqueeze(3)
            res = num/(dem + 1e-16)
            res = res.reshape(X_orig.shape[0], X_orig.shape[1], X_orig.shape[2])
            return res

        Q = self.fc_q(Q)*mask.unsqueeze(-1)
        K, V = self.fc_k(K)*mask.unsqueeze(-1), self.fc_v(K)*mask.unsqueeze(-1)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)

        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = masked_softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V))
        A_n = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        J = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        return Q_, K_, V_, Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), A, A_n, J, O


class Masked_SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(Masked_SAB, self).__init__()
        self.mab = Masked_MAB(dim_in, dim_in, dim_out, num_heads)

    def forward(self, X, mask):
        return self.mab(X, X, mask)

    def forward_check(self, X, mask):
        return self.mab.forward_check(X, X, mask)

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

