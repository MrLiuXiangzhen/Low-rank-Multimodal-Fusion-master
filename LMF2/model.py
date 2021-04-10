from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal

# 在pre-fusion对图像和视频进行处理的网络，实际上只是几个全连接层的叠加，用来提取特征
class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        # 对输入进行BN，Dropout，然后三层全连接层，用的act是relu
        self.norm = nn.BatchNorm1d(in_size)
        # self.cnn = nn.GCN(in_size, hidden_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class  TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in LMF for text
    '''
    # 对于文本输入，用基于LSTM的sub网络进行处理，先过LSTM，然后dropout，最后全连接，这个全连接没有激活函数
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


# 执行Fusion的网络
class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, hidden_dims, text_out, dropouts, output_dim, rank, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()
        # 这里首先记录subnets用到的各种维度
        # dimensions are specified in the order of audio, video and text

        self.video_in = input_dims[0]
        self.text_in = input_dims[1]


        self.video_hidden = hidden_dims[0]
        self.text_hidden = hidden_dims[1]
        self.text_out = text_out
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax


        self.video_prob = dropouts[0]
        self.text_prob = dropouts[1]
        self.post_fusion_prob = dropouts[2]

        # define the pre-fusion subnetworks
        # 这里create三个上面定义的子网络，视频和音频用的是SubNet，文本用的是基于LSTM的TextSubNet，对应论文就是图1的f_v, f_a, f_l, 输出的值就是z_v, z_a, z_l

        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        # 下面定义的层用来在自网络提取了特征之后，乘factors的过程，对应图1中的LMF，每个factor是一个三维的Tensor
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        #self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        #self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        #self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_hidden + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, self.text_out + 1, self.output_dim))
        # 这个fusion_weights的意思是实际上不同的rank之间不是简单的相加，而是有各自的权重，对应172行
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        # 初始化成xavier_normal分布，这个是torch自带的函数

        torch.nn.init.xavier_normal_(self.video_factor)
        torch.nn.init.xavier_normal_(self.text_factor)
        torch.nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, video_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''

        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        batch_size = video_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        # 这里根据图1执行LMF，不过是用高效的方式，简单来说就是交换了求和和按元素相乘的顺序
        if video_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        # 对应图1中额外附加的一个1

        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)


        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)

        # 将三部分的数据经过factor相乘之后的数据相乘，这个乘向量的按元素相乘
        fusion_zy = fusion_video * fusion_text

        
        # output = torch.sum(fusion_zy, dim=0).squeeze(),这里是按照dim=0对fusion的结果求和，注意上面的factor矩阵第一个维度是rank，就是说这里是把不同的rank加起来，对应图1就是图中的若干W相加的过程。
        # use linear transformation instead of simple summation, more flexibility, 具体的实现上是使用了线性的转换而不是简单的相加，这样更有可行性
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output
