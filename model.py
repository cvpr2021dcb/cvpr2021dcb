import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from modeling_bertnew import BertModelNew
from apex import amp
import random

def compute_loss(scores,margin=0.2):

    diagonal = scores.diag().view(scores.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    cost_s = (margin + scores - d1).clamp(min=0)
    cost_im = (margin + scores - d2).clamp(min=0)
    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask).cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)
    eps = 1e-5
    cost_s = cost_s.pow(8).sum(1).add(eps).sqrt().sqrt().sqrt()#.sqrt()#.div(cost_s.size(1)).mul(2)
    cost_im = cost_im.pow(8).sum(0).add(eps).sqrt().sqrt().sqrt()#.sqrt()#.div(cost_im.size(0)).mul(2)
    return cost_s.sum().div(cost_s.size(0)) + cost_im.sum().div(cost_s.size(0))


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def l2norm(X, dim, eps=1e-5):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).add(eps).sqrt() + eps
    X = torch.div(X, norm)
    return X



# RNN Based Language Model





class EncoderText(nn.Module):

    def __init__(self,img_dim, embed_size):
        super(EncoderText, self).__init__()
        self.encoder = BertModelNew.from_pretrained('bert/')
        self.fc = nn.Linear(img_dim,embed_size)

    def forward(self, input_ids, token_type_ids, non_pad_mask, vision_feat, vision_mask, istest=False):
        text_output = self.encoder.embeddings(input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids.long().squeeze())
        text_output = text_output.mul(non_pad_mask.unsqueeze(2).expand(text_output.size()))  

        text_g = text_output.sum(1)

        head_mask = [None]*20
        vision_feat = self.fc(vision_feat)
        vision_feat = self.encoder.embeddings.LayerNorm(vision_feat)

        word_ids = torch.arange(30522, dtype=torch.long).cuda()
        word_all = self.encoder.embeddings.word_embeddings(word_ids)
        word_all = self.encoder.embeddings.LayerNorm(word_all)
        word_all = word_all.permute(1,0)


        vision_feat_new = vision_feat
        vision_g = vision_feat_new.sum(1)


        scores = torch.matmul(vision_feat_new,word_all).mul(20)
        scores = F.softmax(scores,2)
        featnew = torch.matmul(scores,word_all.permute(1,0))
        vision_feat = torch.cat([vision_feat_new,featnew],1)
        vision_mask = torch.cat([vision_mask,vision_mask],1)

        bs = text_output.size(0)
        tl = text_output.size(1)
        vl = vision_feat.size(1)

        extended_attention_mask_text = non_pad_mask[:, None, None, :]
        extended_attention_mask_text = (1.0 - extended_attention_mask_text) * -10000.0

        extended_attention_mask_vision = vision_mask[:, None, None, :]
        extended_attention_mask_vision = (1.0 - extended_attention_mask_vision) * -10000.0


        textnew = self.encoder.encoder(text_output,extended_attention_mask_text,head_mask)
        visionnew = self.encoder.encoder(vision_feat,extended_attention_mask_vision,head_mask) 

        textnew = textnew[0]
        visionnew = visionnew[0]

        text_out = textnew[:,0]
        vision_output =  visionnew.sum(1)

        if istest == False:
            text_out = text_out.unsqueeze(0).expand(bs,-1,-1).contiguous().view(bs*bs,-1)#.view(bs,bs,-1)
            vision_output = vision_output.unsqueeze(1).expand(-1,bs,-1).contiguous().view(bs*bs,-1)
            text_g = text_g.unsqueeze(0).expand(bs,-1,-1).contiguous().view(bs*bs,-1)#.view(bs,bs,-1)
            vision_g = vision_g.unsqueeze(1).expand(-1,bs,-1).contiguous().view(bs*bs,-1)
        else:
            return  vision_output,text_out,vision_g,text_g

        scores =  cosine_similarity(vision_output,text_out,-1)
        scores_g =  cosine_similarity(vision_g,text_g,-1)

        if istest:  
            return scores + scores_g*1
        else:
            scores = scores.view(bs,bs)
            scores_g = scores_g.view(bs,bs)
            return compute_loss(scores) + compute_loss(scores_g)*64



def cosine_similarity(x1, x2, dim=1, eps=1e-5):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)
    
    return similarities


def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities




class DCB(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.txt_enc = EncoderText(2048,768)
        self.txt_enc.cuda()
        cudnn.benchmark = True

        params = list(self.txt_enc.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.txt_enc, self.optimizer = amp.initialize(self.txt_enc, self.optimizer, opt_level= "O1")
        self.txt_enc = torch.nn.DataParallel(self.txt_enc)
        # Loss and Optimizer
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.txt_enc.load_state_dict(state_dict[0])

    def train_start(self):
        """switch to train mode
        """
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.txt_enc.eval()

    def forward_emb(self, images, captions,  target_mask, vision_mask, volatile=False, istest = False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images.float(), volatile=volatile)
        captions = torch.LongTensor(captions)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()#.cuda()
            captions = captions.cuda()#.cuda()
        # Forward
        n_img = images.size(0)
        n_cap = captions.size(0)

        attention_mask = get_non_pad_mask(captions).cuda().squeeze()
        token_type_ids = torch.zeros_like(attention_mask)

        video_non_pad_mask = get_non_pad_mask(vision_mask).cuda().squeeze()

        scores = self.txt_enc(captions, token_type_ids, attention_mask,images,video_non_pad_mask,istest)
        return scores


 
    def forward_loss(self, img_emb,cap_emb, cap_len, text_non_pad_mask, text_slf_attn_mask,img_non_pad_mask,img_slf_attn_mask, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        scores = self.cross_att(img_emb,cap_emb,cap_len,text_non_pad_mask, text_slf_attn_mask,img_non_pad_mask,img_slf_attn_mask)
        loss = self.criterion(scores)
        self.logger.update('Le', loss.item(), scores.size(0))
        return loss

    def train_emb(self, images, captions, target_mask, vision_mask, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
         
        # measure accuracy and record loss
        scores = self.forward_emb(images, captions, target_mask, vision_mask)
        # measure accuracy and record loss

        self.optimizer.zero_grad()
        if scores is not None:
           loss = scores.sum()
           self.logger.update('Le', loss, images.size(0))
        else:
           return
        # compute gradient and do SGD step
        #loss.backward()
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
           scaled_loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()



