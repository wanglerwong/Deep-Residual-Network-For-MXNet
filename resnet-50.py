'''
Deep Residual Learning for Image Recognition, http://arxiv.org/abs/1512.03385

# write by Huang, 5 Apr 2016
'''
import mxnet as mx
import find_mxnet

def conv_factory(data, num_filter, kernel, stride, pad, act_type='relu', conv_type=0):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)   
    bn = mx.symbol.BatchNorm(data=conv)
    if conv_type == 0:
        act = mx.symbol.Activation(data=bn, act_type=act_type)
        return act
    elif conv_type == 1:
        return bn

def res_factory(data=data, num_filter=num_filter, dim_match='True'):
    if dim_match == True:
        identity_data = data
        conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
        conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        conv3 = conv_factory(data=conv2, num_filter=4*num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), conv_type=1)
        new_data = identity_data + conv3
        act = mx.symbol.Activation(data=new_data, act_type='relu')
        return act
    elif:
        project_data = conv_factory(data=data, num_filter=4*num_filter, kernel=(1, 1), stride=(2, 2), pad=(0, 0), conv_type=1)
        conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
        conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3, 3), stride=(2, 2), pad=(1, 1))
        conv3 = conv_factory(data=conv2, num_filter=4*num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), conv_type=1)
        new_data = project_data + conv3
        act = mx.symbol.Activation(data=new_data, act_type='relu')
        return act
        
def res_net(data, num1, num2, num3, num4):
    for i in xrange(num1):
        data = res_factory(data=data, num_filter=64)

    for i in xrange(num2):
        if i == 0:
            data = res_factory(data=data, num_filter=128, dim_match='False')
        else:
            data = res_factory(data=data, num_filter=128)

    for i in xrange(num3):
        if i == 0:
            data = res_factory(data=data, num_filter=256, dim_match='False')
        else:
            data = res_factory(data=data, num_filter=256)

    for i in xrange(num4):
        if i == 0:
            data = res_factory(data=data, num_filter=512, dim_match='False')
        else:
            data = res_factory(data=data, num_filter=512)

    return data

def get_symbol(num_class=1000):
    conv = conv_factory(data=mx.symbol.Variable(name='data'), num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3))
    pool1 = mx.symbol.Pooling(data=conv, kernel=(3, 3), stride=(2, 2), pool_type='max')
    
    resnet = res_net(data=pool, num1=3, num2=4, num3=6, num4=3)

    pool2 = mx.symbol.Pooling(data=resnet, kernel=(7, 7), stride=(1, 1), pool_type='avg')
    flatten = mx.symbol.Flatten(data=pool2, name='flatten')
    fc = mx.symbol.FullConnected(data=flatten, num_hidden=num_class, name='fc')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax

