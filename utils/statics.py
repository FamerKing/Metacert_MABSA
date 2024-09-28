import torch.nn as nn
import torch
import torch.nn.functional as F
from TorchCRF import CRF
from torch.nn import CrossEntropyLoss
import torchvision
import numpy as np
import os
from transformers import RobertaModel, BertModel, AlbertModel, ElectraModel, ViTModel, SwinModel, DeiTModel, ConvNextModel


class DTCAModel(nn.Module):
    def __init__(self, config1, config2, text_num_labels, alpha, beta, gamma, text_model_name="roberta",
                 image_model_name='vit'):
        super().__init__()
        if text_model_name == 'roberta':
            self.roberta = RobertaModel(config1, add_pooling_layer=False)
        elif text_model_name == 'bert':
            self.bert = BertModel(config1, add_pooling_layer=False)
        elif text_model_name == 'albert':
            self.albert = AlbertModel(config1, add_pooling_layer=False)
        elif text_model_name == 'electra':
            self.electra = ElectraModel(config1)
        if image_model_name == 'vit':
            self.vit = ViTModel(config2)
        elif image_model_name == 'swin':
            self.swin = SwinModel(config2)
        elif image_model_name == 'deit':
            self.deit = DeiTModel(config2)
        elif image_model_name == 'convnext':
            self.convnext = ConvNextModel(config2)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.config1 = config1
        self.config2 = config2
        self.text_num_labels = text_num_labels
        self.image_text_cross = MultiHeadAttention(8, config1.hidden_size, config1.hidden_size, config1.hidden_size)
        self.dropout = nn.Dropout(config1.hidden_dropout_prob)
        self.loss_fct = CrossEntropyLoss()
        self.classifier1 = nn.Linear(config1.hidden_size, self.text_num_labels)
        self.classifier0 = nn.Linear(config1.hidden_size, self.text_num_labels)
        self.CRF = CRF(self.text_num_labels, batch_first=True)

        # 温度缩放
        self.temperature_scaling = TemperatureScaling()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                pixel_values=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                image_labels=None,
                itm_l=None,
                head_mask=None,
                cross_labels=None,
                return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config1.use_return_dict
        if self.text_model_name == 'bert':
            text_outputs = self.bert(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     inputs_embeds=inputs_embeds,
                                     output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states,
                                     return_dict=return_dict)
        elif self.text_model_name == 'roberta':
            text_outputs = self.roberta(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        elif self.text_model_name == 'albert':
            text_outputs = self.albert(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       head_mask=head_mask,
                                       inputs_embeds=inputs_embeds,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states,
                                       return_dict=return_dict)
        elif self.text_model_name == 'electra':
            text_outputs = self.electra(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        else:
            text_outputs = None
        if self.image_model_name == 'vit':
            image_outputs = self.vit(pixel_values, head_mask=head_mask)
        elif self.image_model_name == 'swin':
            image_outputs = self.swin(pixel_values, head_mask=head_mask)
        elif self.image_model_name == 'deit':
            image_outputs = self.deit(pixel_values, head_mask=head_mask)
        elif self.image_model_name == 'convnext':
            image_outputs = self.convnext(pixel_values)
        else:
            image_outputs = None

        # itm 动态控制贡献程度
        if itm_l is not None:
            count_nonzero = torch.count_nonzero(itm_l).item()  # 统计非零元素的个数
            sum_of_elements = itm_l.numel()  # 计算所有元素的个数
            delta = count_nonzero / sum_of_elements

        text_last_hidden_states = text_outputs["last_hidden_state"]
        image_last_hidden_states = image_outputs["last_hidden_state"]

        # cross_crf_loss
        image_text_cross_attention, _ = self.image_text_cross(text_last_hidden_states, image_last_hidden_states,
                                                              image_last_hidden_states)
        cross_logits = self.classifier0(image_text_cross_attention)
        mask = (labels != -100)
        mask[:, 0] = 1
        cross_crf_loss = -self.CRF(cross_logits, cross_labels, mask=mask) / 10

        # 温度缩放应用
        scaled_logits = self.temperature_scaling(cross_logits)

        # 转化为概率分布
        text_probs = F.softmax(text_last_hidden_states, dim=-1)
        image_probs = F.softmax(image_last_hidden_states, dim=-1)

        # 计算KL散度
        kl_div_loss = kl_divergence_loss(text_probs, image_probs, mask)

        # text_loss
        sequence_output1 = self.dropout(text_last_hidden_states)
        text_token_logits = self.classifier1(sequence_output1)

        text_loss = self.loss_fct(text_token_logits.view(-1, self.text_num_labels), labels.view(-1))

        if delta == 0:
            loss = text_loss
        else:
            loss = cross_crf_loss + self.gamma * kl_div_loss + self.alpha * text_loss

        return {"loss": loss,
                "logits": text_token_logits,
                "cross_logits": scaled_logits}


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False, attn_type='softmax'):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if dropout2:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout2)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, attn_mask=None, dec_self=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        if hasattr(self, 'dropout2'):
            q = self.dropout2(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        if hasattr(self, 'fc'):
            output = self.fc(output)

        if hasattr(self, 'dropout'):
            output = self.dropout(output)

        if dec_self:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e6)

        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def kl_divergence_loss(p_logits, q_logits, mask=None):
    '''
    计算概率分布 p 和 q 之间的KL散度。
    :param p_logits: [N, *, L] 原始分布的 logits。
    :param q_logits: [N, *, L] 目标分布的 logits。
    :param mask: [N] 可选的权重或掩码，用于指示有效样本。
    '''
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)

    # 计算KL散度，此时不使用reduction参数，保持输出形状与输入形状一致
    kl_div = F.kl_div(p_log_probs, q_probs, reduction='none')

    # 在样本维度进行求和
    kl_div = kl_div.sum(dim=-1).sum(dim=-1)

    # 应用`mask`以仅选择有效的样本
    if mask is not None:
        mask = mask.to(dtype=kl_div.dtype)
        mask = mask.unsqueeze(-1)  # 扩展mask的形状为[N, 1]使其能够与kl_div进行广播
        kl_div = kl_div * mask

        # 计算使用`mask`加权平均后的损失
        kl_div_loss = kl_div.sum() / mask.sum()
    else:
        # 如果没有`mask`，则直接计算平均损失
        kl_div_loss = kl_div.mean()

    return kl_div_loss


class TemperatureScaling(nn.Module):
    def __init__(self, init_temp=1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits):
        return logits / self.temperature
