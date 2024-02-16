import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from opt_einsum import contract
import os
from transformers import BertTokenizer
from transformers import BertConfig, BertModel, AdamW
from main import generate_parser


parser = generate_parser()
args = parser.parse_args()

# # 设置使用的GPU设备
# ngpus_per_node = torch.cuda.device_count()
# print('gpu数量: ', ngpus_per_node)
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cuda:1'
# print(device)

model_path = args.model_path
device = args.device
tokenizer = BertTokenizer.from_pretrained(model_path)
# model_config = BertConfig.from_pretrained(model_path)
# label_model = BertModel.from_pretrained(model_path, config=model_config).to('cpu')

def fusion(input):
    b_size,nul = input.size(0),input.size(1)
    l = nn.Linear(nul,b_size).to(device)
    return l

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, queries, keys, values, mask=None):
        b, n, _, h = queries.size(0),queries.size(1), queries.size(2), self.heads
        ueries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)
        queries = ueries.view(b, n, h, -1).transpose(1, 2)
        keys = keys.view(b, n, h, -1).transpose(1, 2)
        values = values.view(b, n, h, -1).transpose(1, 2)
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, n, -1)
            dots.masked_fill_(~mask, float('-inf'))
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, values)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return out

class ClassifiyModule(nn.Module):
    def __init__(self, Bertmodel_path, Bertmodel_config):
        super(ClassifiyModule, self).__init__()
        self.PreBert = BertModel.from_pretrained(Bertmodel_path, config=Bertmodel_config)
        #         # 冻结参数
        #         for para in self.PreBert.parameters():
        #           para.requires_grad = False
        self.maxPool2d = nn.AdaptiveMaxPool2d((1, 1024))
        self.dropout = nn.Dropout(p=0.1)
        self.classify = nn.Linear(1024, 146)
        self.size = 1024
        self.d_a = 256
        self.n_labels = 146

        self.cross_att = CrossAttention(1024)

        self.dropout_laat = nn.Dropout(0.2)

        self.first_linears = nn.Linear(self.size, self.d_a, bias=False)
        self.second_linears = nn.Linear(self.d_a, self.n_labels, bias=False)
        self.third_linears = nn.Linear(self.size,self.n_labels, bias=True)

        self.w_linear = nn.Linear(self.size,self.size,bias=True)
        self.b_linear = nn.Linear(self.size,1,bias=True)

        self.fusion = nn.Linear(self.n_labels,args.batch_size)

        self.nor = torch.nn.LayerNorm(1024, eps=1e-05, elementwise_affine=True)

        # 使用match
        self.match_third_linears = nn.Linear(512, self.n_labels, bias=True)

        self.one = nn.Linear(2, 1)
        self.maxPool2d = nn.AdaptiveMaxPool2d((1, 1024))

    def forward(self, input_ids, token_type_ids, attention_mask, return_dict,label_fea,true_splitNumbers,chief):

        label_fea = label_fea.max(0)[0]
        input_ids = input_ids.view(-1, 512)  # input_ids.shape=(batch_size*max_splitNumbers,seq_len)
        attention_mask = attention_mask.view(-1, 512)  # input_ids.shape=(batch_size*max_splitNumbers,seq_len)
        x = self.PreBert(input_ids,
                         token_type_ids=token_type_ids,
                         attention_mask=attention_mask,
                         return_dict=return_dict)  # ((batch_size*max_splitNumbers,1024),(batch_size*max_splitNumbers,1024))

        # word_num = x[0].size(1)
        # logits_pool = x[0]

        # # 计算注意力权重，这里使用均匀分布的权重，你可以根据需要使用不同的方法计算权重
        # attention_weights = F.softmax(self.one.weight.T, dim=0)
        # # 使用注意力权重加权平均，将第二个维度聚合为一个维度
        # # x1 = torch.sum(x[1].reshape(-1, 2, 1024) * attention_weights.view(1, 2, 1), dim=1)
        # logits_pool = torch.sum(x[0].reshape(-1, 2, 512, 1024) * attention_weights.view(1, 2, 1, 1), dim=1)

        # temp = []
        # logits_pool_w = x[1].view(-1, 2, 1024)
        # for i in range(len(true_splitNumbers)):
        #     temp.append(self.maxPool2d(logits_pool_w[i][0:true_splitNumbers[i]].view(1, 1, -1, 1024)).view(1, 1024))
        # logits_pool_w = torch.cat(temp, dim=0)


        temp = []
        logits_pool_w = x[0].view(-1, 2, 512, 1024)
        for i in range(len(true_splitNumbers)):
            temp.append(logits_pool_w[i][0:true_splitNumbers[i]].view(1, -1, 512, 1024).max(1)[0])
        logits_pool = torch.cat(temp, dim=0)

        #融合chief和text做一个交互
        xx_chief_text = self.cross_att(chief,logits_pool,logits_pool)
        logits_pool = xx_chief_text + logits_pool


        # # 使用acl2023匹配(GPU)
        # match_score = contract('abc,ebc->aebc', logits_pool, label_fea)
        # # print("match_score.size()",match_score.size())
        # # 找到倒数第二大的值
        # sorted_tensor, _ = torch.sort(match_score, dim=-1)
        # second_max_values = sorted_tensor[..., -2]
        # # 找到最大值
        # max_values, _ = torch.max(match_score, dim=-1)
        # # 计算新的最后一维的值
        # new_last_dim = 2 * max_values - second_max_values
        # # print("new_last_dim.size()",new_last_dim.size())
        # match_weighted_output = self.match_third_linears.weight.mul(new_last_dim).sum(dim=2).add(
        #     self.match_third_linears.bias)



        #使用注意力来融合所有的单词
        t = self.dropout_laat(logits_pool)
        score = F.softmax(contract('abc,abc->abc',t,t),dim=1)
        m = contract('abc,abc->abc',t,score)
        m = m + t


        weights = torch.tanh(self.first_linears(m))
        att_weights = self.second_linears(weights)
        att_weights = F.softmax(att_weights, 1).transpose(1, 2)
        weighted_output = att_weights @ m
        weighted_output = self.third_linears.weight.mul(weighted_output).sum(dim=2).add(self.third_linears.bias)

        # x = weighted_output + match_weighted_output
        # x = fusion(match_weighted_output).weight.mul(match_weighted_output) + x

        # # # laat使用的是全部的字级别的表示
        # # x = logits_pool.max(1)[0].view(-1, 1024)  # (batch_size,max_splitNumbers,1024)
        # x = self.dropout(x[1])  # (batch,1024)
        # x = self.classify(x)  # (batch,148)

        x = weighted_output

        return x


class ClassifiyLSTM(nn.Module):
    def __init__(self, Bertmodel_path, Bertmodel_config):
        super(ClassifiyLSTM, self).__init__()
        self.PreBert = BertModel.from_pretrained(Bertmodel_path, config=Bertmodel_config)

        self.dropout = nn.Dropout(0.2)
        self.bidirectional = True
        self.n_layers = 1
        self.hidden_dim=512
        self.rnn = nn.LSTM(1024, self.hidden_dim, num_layers=1,
                           bidirectional=self.bidirectional, dropout=0.2,batch_first=True)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        number = 1
        if self.bidirectional:
            number = 2

        hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
                  weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda()
                  )
        return hidden

    def forward(self, inputs_ids, attention_mask, true_splitnumbers):
        inputs_ids = inputs_ids.view(-1,512)
        attention_mask = attention_mask.view(-1,512)
        # print("input",inputs_ids.size())

        x = self.PreBert(inputs_ids,
                         token_type_ids=None,
                         attention_mask=attention_mask,
                         return_dict=False)

        lstm_input = x[0]
        hidden = self.init_hidden(lstm_input.size(0))

        lstm_out, (hidden_last, cn_last) = self.rnn(lstm_input, hidden)
        # 修改 双向的需要单独处理
        if self.bidirectional:
            # 正向最后一层，最后一个时刻
            hidden_last_L = hidden_last[-2]
            # print(hidden_last_L.shape)  #[32, 384]
            # 反向最后一层，最后一个时刻
            hidden_last_R = hidden_last[-1]
            # print(hidden_last_R.shape)   #[32, 384]
            # 进行拼接
            # hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
            hidden_last_out = hidden_last_L + hidden_last_R
            # print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out = hidden_last[-1]  # [32, 384]
        lstm_out = lstm_out.reshape(lstm_input.size(0),-1,512,1024)
        # print('lstm', lstm_out.size())
        lstm_out = torch.max(lstm_out,dim=1)[0]
        # print('lstm1',lstm_out.size())
        out = self.dropout(lstm_out)

        return out


