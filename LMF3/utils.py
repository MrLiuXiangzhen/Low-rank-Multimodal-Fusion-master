from torch.utils.data import Dataset
import sys
import numpy as np

# 根据版本信息，读取不同的pickle
if sys.version_info.major == 2:
    import cPickle as pickle
else:
    import pickle

import pdb

AUDIO = b'covarep'
VISUAL = b'facet'
TEXT = b'glove'
LABEL = b'label'
TRAIN = b'train'
VALID = b'valid'
TEST = b'test'

# 统计输入了多少超参数
def total(params):
    '''
    count the total number of hyperparameter settings
    '''
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings


# 读取pom数据集
def load_pom(data_path):
    # parse the input args
    class POM(Dataset):
        '''
        PyTorch Dataset for POM, don't need to change this
        '''
        # pytorch标准的读取数据方式，实现三个方法，torch会自动调用getitem来读取数据
        def __init__(self, audio, visual,  labels):
            self.audio = audio
            self.visual = visual
            
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    # 用pickle读取数据
    if sys.version_info.major == 2:
        pom_data = pickle.load(open(data_path + "pom.pkl", 'rb'))
    else:
        pom_data = pickle.load(open(data_path + "pom.pkl", 'rb'), encoding='bytes')
    # 切分训练、验证和测试集
    pom_train, pom_valid, pom_test = pom_data[TRAIN], pom_data[VALID], pom_data[TEST]

    # 数据读出来是一个字典，这里取出数据中的不同部分，audio，video和text
    train_audio, train_visual, train_labels \
        = pom_train[AUDIO], pom_train[VISUAL],  pom_train[LABEL]
    valid_audio, valid_visual, valid_labels \
        = pom_valid[AUDIO], pom_valid[VISUAL], pom_valid[LABEL]
    test_audio, test_visual,  test_labels \
        = pom_test[AUDIO], pom_test[VISUAL], pom_test[LABEL]

    # code that instantiates the Dataset objects
    train_set = POM(train_audio, train_visual,  train_labels)
    valid_set = POM(valid_audio, valid_visual,  valid_labels)
    test_set = POM(test_audio, test_visual,  test_labels)


    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    
    input_dims = (audio_dim, visual_dim)

    # remove possible NaN values
    # 删除Nan，因为np.nan == np.nan返回的是False，所以nan的数据在判断中是true，会被设置成0
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    
    
    print(np.any(np.isnan(train_set.visual)))
    print(np.any(np.isnan(train_set.audio)))
    

    return train_set, valid_set, test_set, input_dims

def load_iemocap(data_path, emotion):
    # parse the input args
    class IEMOCAP(Dataset):
        '''
        PyTorch Dataset for IEMOCAP, don't need to change this
        '''
        def __init__(self, audio, visual,  labels):
            self.audio = audio
            self.visual = visual
            
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :],  self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    if sys.version_info.major == 2:
        iemocap_data = pickle.load(open(data_path + "iemocap.pkl", 'rb'))
    else:
        iemocap_data = pickle.load(open(data_path + "iemocap.pkl", 'rb'), encoding='bytes')
    iemocap_train, iemocap_valid, iemocap_test = iemocap_data[emotion][TRAIN], iemocap_data[emotion][VALID], iemocap_data[emotion][TEST]

    train_audio, train_visual, train_labels \
        = iemocap_train[AUDIO], iemocap_train[VISUAL],  iemocap_train[LABEL]
    valid_audio, valid_visual, valid_labels \
        = iemocap_valid[AUDIO], iemocap_valid[VISUAL], iemocap_valid[LABEL]
    test_audio, test_visual,  test_labels \
        = iemocap_test[AUDIO], iemocap_test[VISUAL], iemocap_test[LABEL]

    # code that instantiates the Dataset objects
    train_set = IEMOCAP(train_audio, train_visual, train_labels)
    valid_set = IEMOCAP(valid_audio, valid_visual,  valid_labels)
    test_set = IEMOCAP(test_audio, test_visual, test_labels)


    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    
    input_dims = (audio_dim, visual_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    

    return train_set, valid_set, test_set, input_dims

def load_mosi(data_path):

    # parse the input args
    class MOSI(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, labels):
            self.audio = audio
            self.visual = visual
            
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :],  self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    if sys.version_info.major == 2:
        mosi_data = pickle.load(open(data_path + "mosi.pkl", 'rb'))
    else:
        mosi_data = pickle.load(open(data_path + "mosi.pkl", 'rb'), encoding='bytes')
    mosi_train, mosi_valid, mosi_test = mosi_data[TRAIN], mosi_data[VALID], mosi_data[TEST]

    train_audio, train_visual, train_labels \
        = mosi_train[AUDIO], mosi_train[VISUAL], mosi_train[LABEL]
    valid_audio, valid_visual, valid_labels \
        = mosi_valid[AUDIO], mosi_valid[VISUAL], mosi_valid[LABEL]
    test_audio, test_visual,  test_labels \
        = mosi_test[AUDIO], mosi_test[VISUAL],  mosi_test[LABEL]

    print(train_audio.shape)
    print(train_visual.shape)
    
    print(train_labels.shape)

    # code that instantiates the Dataset objects
    train_set = MOSI(train_audio, train_visual,  train_labels)
    valid_set = MOSI(valid_audio, valid_visual,  valid_labels)
    test_set = MOSI(test_audio, test_visual,  test_labels)


    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    
    input_dims = (audio_dim, visual_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0
    
    
    
    print(np.any(np.isnan(train_set.visual)))
    print(np.any(np.isnan(train_set.audio)))
    

    return train_set, valid_set, test_set, input_dims
