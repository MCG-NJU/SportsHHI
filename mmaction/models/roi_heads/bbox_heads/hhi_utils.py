import torch
import torch.nn as nn
import copy


def my_minmax(input: torch.Tensor, zero_mat: torch.Tensor, one_mat: torch.Tensor) -> torch.Tensor:
    temp_mat = torch.stack([input, zero_mat]).max(0).values
    ret = torch.stack([temp_mat, one_mat]).min(0).values
    return ret


def boxes2union(box1: torch.Tensor, box2: torch.Tensor, pooling_size: int) -> torch.Tensor:
    """generate spatial masks for bboxes
    modified from https://github.com/rowanz/neural-motifs/blob/master/lib/draw_rectangles/draw_rectangles.pyx

    Args:
        box1 (torch.Tensor): tensor of p1 bboxes
        box2 (torch.Tensor): tensor of p2 bboxes
        pooling_size (int): size of spatial mask

    Returns:
        torch.Tensor: spatial masks
    """
    assert box1.shape == box2.shape
    n, _ = box1.shape

    x1_union = torch.stack((box1[:,0], box2[:,0]), 1).min(1).values
    y1_union = torch.stack((box1[:,1], box2[:,1]), 1).min(1).values
    x2_union = torch.stack((box1[:,2], box2[:,2]), 1).max(1).values
    y2_union = torch.stack((box1[:,3], box2[:,3]), 1).max(1).values

    w = x2_union - x1_union
    h = y2_union - y1_union

    x_mat = torch.arange(pooling_size).repeat(pooling_size, 1)
    x_mat = x_mat.repeat(n, 1, 1).float().to(box1.device)
    y_mat = torch.arange(pooling_size).reshape(-1,1).repeat(1,pooling_size)
    y_mat = y_mat.repeat(n,1,1).float().to(box1.device)
    zero_mat = torch.zeros(x_mat.shape).to(box1.device)
    one_mat = torch.ones(x_mat.shape).to(box1.device)

    x1_box_1 = ((box1[:,0] - x1_union) * pooling_size / w).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    y1_box_1 = ((box1[:,1] - y1_union) * pooling_size / h).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    x2_box_1 = ((box1[:,2] - x1_union) * pooling_size / w).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    y2_box_1 = ((box1[:,3] - y1_union) * pooling_size / h).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    
    x_contrib_1 = my_minmax(x_mat+one_mat-x1_box_1,zero_mat,one_mat) * my_minmax(x2_box_1-x_mat,zero_mat,one_mat)
    y_contrib_1 = my_minmax(y_mat+one_mat-y1_box_1,zero_mat,one_mat) * my_minmax(y2_box_1-y_mat,zero_mat,one_mat)

    x1_box_2 = ((box2[:,0] - x1_union) * pooling_size / w).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    y1_box_2 = ((box2[:,1] - y1_union) * pooling_size / h).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    x2_box_2 = ((box2[:,2] - x1_union) * pooling_size / w).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    y2_box_2 = ((box2[:,3] - y1_union) * pooling_size / h).reshape(-1,1).repeat(1,pooling_size*pooling_size).reshape(-1, pooling_size, pooling_size)
    x_contrib_2 = my_minmax(x_mat+one_mat-x1_box_2,zero_mat,one_mat) * my_minmax(x2_box_2-x_mat,zero_mat,one_mat)
    y_contrib_2 = my_minmax(y_mat+one_mat-y1_box_2,zero_mat,one_mat) * my_minmax(y2_box_2-y_mat,zero_mat,one_mat)

    return torch.stack([x_contrib_1*y_contrib_1,x_contrib_2*y_contrib_2]).transpose(1,0)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim=1792, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        # self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, input_key_padding_mask):
        # local attention
        src2, local_attention_weights = self.self_attn(
            src, src, src, key_padding_mask=input_key_padding_mask)
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        # src = self.norm2(src)

        return src, local_attention_weights


class TransformerEncoder(nn.Module):
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, input, input_key_padding_mask):
        output = input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)
        for i, layer in enumerate(self.layers):
            output, local_attention_weihts = layer(output, input_key_padding_mask)
            weights[i] = local_attention_weihts
        if self.num_layers > 0:
            return output, weights
        else:
            return output, None
        

class transformer(nn.Module):
    
    def __init__(self, enc_layer_num=1, embed_dim=1792, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder_layers = TransformerEncoder(encoder_layer, enc_layer_num)

    def forward(self, features, im_idx):
        l = torch.sum(im_idx == torch.mode(im_idx)[0])
        b = int(im_idx[-1] + 1)
        rel_input = torch.zeros([l, b, features.shape[1]]).to(features.device)
        masks = torch.zeros([b, l], dtype=bool).to(features.device)
        for i in range(b):
            rel_input[:torch.sum(im_idx==i), i, :] = features[im_idx==i]
            masks[i, torch.sum(im_idx==i):] = True

        enc_output, attention_weights = self.encoder_layers(rel_input, masks)
        enc_output = (enc_output.permute(1,0,2)).contiguous().view(-1, features.shape[1])[masks.view(-1)==0]

        return enc_output, attention_weights