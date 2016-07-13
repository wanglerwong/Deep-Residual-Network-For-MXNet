'''
Inception-ResNet-v2

Reference:
	Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning

By huangxuankun, 2016.07.06.
'''

import mxnet as mx

def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix='', flag=0):
	conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
	bn = mx.symbol.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
	if flag == 0:
		act = mx.symbol.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
		return act
	else:
		return bn

def Stem(data, suffix):
	conv1  = Conv(data, 32, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name=('%s_conv1'%suffix)) # 149*149*32
	conv2  = Conv(conv1, 32, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name=('%s_conv2'%suffix)) # 147*147*32
	conv3  = Conv(conv2, 64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=('%s_conv3'%suffix)) # 147*147*64
	
	maxpool4  = mx.symbol.Pooling(data=conv3, kernel=(3, 3), stride=(2, 2), pool_type='max', name=('%s_maxpool4'%suffix)) # 73*73*64
	conv4  = Conv(conv3, 96, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name=('%s_conv4'%suffix)) # 73*73*96
	
	concat1 = mx.symbol.Concat(*[maxpool4, conv4], name=('%s_concat1'%suffix)) 

	conv5 = Conv(concat1, 64, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=('%s_conv51'%suffix)) # 73*73*64
	conv5 = Conv(conv5, 96, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name=('%s_conv52'%suffix)) # 71*71*96

	conv6 = Conv(concat1, 64, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=('%s_conv61'%suffix)) # 73*73*64
	conv6  = Conv(conv6, 64, kernel=(7, 1), stride=(1, 1), pad=(3, 0), name=('%s_conv62'%suffix)) # 73*73*64
	conv6  = Conv(conv6, 64, kernel=(1, 7), stride=(1, 1), pad=(0, 3), name=('%s_conv63'%suffix)) # 73*73*64
	conv6  = Conv(conv6, 96, kernel=(3, 3), stride=(1, 1), pad=(0, 0), name=('%s_conv64'%suffix)) # 71*71*96
	
	concat2 = mx.symbol.Concat(*[conv5, conv6], name=('%s_concat2'%suffix)) # 71*71*192

	maxpool7 = mx.symbol.Pooling(data=concat2, kernel=(3, 3), stride=(2, 2), pool_type='max', name=('%s_maxpool7'%suffix)) # 35*35*192
	conv7 = Conv(concat2, 192, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name=('%s_conv7'%suffix)) # 35*35*192

	concat3 = mx.symbol.Concat(*[maxpool7, conv7], name=('%s_concat3'%suffix))
	out = mx.symbol.BatchNorm(data=concat3, fix_gamma=True, name=('%s_batchnorm3'%suffix))
	out = mx.symbol.Activation(data=out, act_type='relu', name=('%s_relu3'%suffix))
	return out


def InceptionResnetA(data, suffix):
	conv1  = Conv(data, 32, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=('%s_conv11'%suffix)) # 35*35*32
	
	conv2  = Conv(data, 32, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=('%s_conv21'%suffix)) # 35*35*32
	conv2  = Conv(conv2, 32, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=('%s_conv22'%suffix)) # 35*35*32
	
	conv3  = Conv(data, 32, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=('%s_conv31'%suffix)) # 35*35*32
	conv3  = Conv(conv3, 48, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=('%s_conv32'%suffix)) # 35*35*48
	conv3  = Conv(conv3, 64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=('%s_conv33'%suffix)) # 35*35*64
	
	concat1 = mx.symbol.Concat(*[conv1, conv2, conv3], name=('%s_concat1'%suffix))
	conv4 = Conv(concat1, 384, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=('%s_conv4'%suffix))

	add = data + conv4
	# out = mx.symbol.BatchNorm(data=add, fix_gamma=True, name=('%s_out1'%suffix))
	# out = mx.symbol.Activation(data=out, act_type='relu', name=('%s_out2'%suffix))
	return add


def ReductionA(data, suffix):
	maxpool  = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type='max', name=('%s_maxpool'%suffix)) # 17*17
	
	conv1  = Conv(data, 384, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name=('%s_conv111'%suffix)) # 17*17*384
	
	conv2  = Conv(data, 256, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=('%s_conv211'%suffix)) # 35*35*256
	conv2  = Conv(conv2, 256, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=('%s_conv212'%suffix)) # 35*35*256
	conv2  = Conv(conv2, 384, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name=('%s_conv213'%suffix)) # 17*17*384
	
	concat = mx.symbol.Concat(*[maxpool, conv1, conv2], name=('%s_concat1'%suffix))
	out = mx.symbol.BatchNorm(data=concat, fix_gamma=True, name=('%s_out1'%suffix))
	out = mx.symbol.Activation(data=out, act_type='relu', name=('%s_out2'%suffix))
	return out

def InceptionResnetB(data, suffix=''):
	conv1 = Conv(data, 192, kernel=(1, 1), stride=(1, 1), pad=(0,0), name=('%s_conv1'%suffix)) # 17*17*192

	conv2 = Conv(data, 120, kernel=(1, 1), stride=(1, 1), pad=(0,0), name=('%s_conv21'%suffix)) # 17*17*120
	conv2 = Conv(conv2, 160, kernel=(1, 7), stride=(1, 1), pad=(0,3), name=('%s_conv22'%suffix)) # 17*17*160
	conv2 = Conv(conv2, 192, kernel=(7, 1), stride=(1, 1), pad=(3,0), name=('%s_conv23'%suffix)) # 17*17*192

	concat1 = mx.symbol.Concat(*[conv1, conv2], name=('%s_concat1'%suffix))
	conv3 = Conv(concat1, 1152, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=('%s_conv3'%suffix))

	add = data + conv3
	# out = mx.symbol.BatchNorm(data=add, fix_gamma=True, name=('%s_out1'%suffix))
	# out = mx.symbol.Activation(data=out, act_type='relu', name=('%s_out2'%suffix))
	return add

def ReductionB(data, suffix):
	maxpool  = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type='max', name=('%s_maxpool'%suffix)) # 8*8
	
	conv1  = Conv(data, 256, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=('%s_conv11'%suffix)) # 17*17*256
	conv1  = Conv(conv1, 384, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name=('%s_conv12'%suffix)) # 8*8*384

	conv2  = Conv(data, 256, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=('%s_conv21'%suffix)) # 17*17*256
	conv2  = Conv(conv2, 288, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name=('%s_conv22'%suffix)) # 8*8*288
	
	conv3  = Conv(data, 256, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=('%s_conv31'%suffix)) # 17*17*256
	conv3  = Conv(conv3, 288, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=('%s_conv32'%suffix)) # 17*17*288
	conv3  = Conv(conv3, 320, kernel=(3, 3), stride=(2, 2), pad=(0, 0), name=('%s_conv33'%suffix)) # 8*8*320
	
	concat = mx.symbol.Concat(*[maxpool, conv1, conv2, conv3], name=('%s_concat1'%suffix))
	out = mx.symbol.BatchNorm(data=concat, fix_gamma=True, name=('%s_out1'%suffix))
	out = mx.symbol.Activation(data=out, act_type='relu', name=('%s_out2'%suffix))
	return out

def InceptionResnetC(data, suffix):
	conv1 = Conv(data, 192, kernel=(1, 1), stride=(1, 1), pad=(0,0), name=('%s_conv1'%suffix)) # 8*8*192

	conv2 = Conv(data, 192, kernel=(1, 1), stride=(1, 1), pad=(0,0), name=('%s_conv21'%suffix)) # 8*8*192
	conv2 = Conv(conv2, 224, kernel=(1, 3), stride=(1, 1), pad=(0,1), name=('%s_conv22'%suffix)) # 8*8*224
	conv2 = Conv(conv2, 256, kernel=(3, 1), stride=(1, 1), pad=(1,0), name=('%s_conv23'%suffix)) # 8*8*256

	concat1 = mx.symbol.Concat(*[conv1, conv2], name=('%s_concat1'%suffix))
	conv3 = Conv(concat1, 2144, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=('%s_conv3'%suffix))

	add = data + conv3
	# out = mx.symbol.BatchNorm(data=add, fix_gamma=True, name=('%s_out1'%suffix))
	# out = mx.symbol.Activation(data=out, act_type='relu', name=('%s_out2'%suffix))
	return add


def get_symbol(num_classes=1000):
	data = mx.symbol.Variable(name='data') # Output: 299*299
	# stage 1
	st = Stem(data, 'stem') # Output: 35*35
	# stage 2
	in2a = InceptionResnetA(st, 'in2a') # Output: 35*35
	in2b = InceptionResnetA(in2a, 'in2b') # Output: 35*35
	in2c = InceptionResnetA(in2b, 'in2c') # Output: 35*35
	in2d = InceptionResnetA(in2c, 'in2d') # Output: 35*35
	in2e = InceptionResnetA(in2d, 'in2e') # Output: 35*35
	rd2 = ReductionA(in2e, 'rd2') # Output: 17*17
	# stage 3
	in3a = InceptionResnetB(rd2, 'in3a') # Output: 17*17
	in3b = InceptionResnetB(in3a, 'in3b') # Output: 17*17
	in3c = InceptionResnetB(in3b, 'in3c') # Output: 17*17
	in3d = InceptionResnetB(in3c, 'in3d') # Output: 17*17
	in3e = InceptionResnetB(in3d, 'in3e') # Output: 17*17
	in3f = InceptionResnetB(in3e, 'in3f') # Output: 17*17
	in3g = InceptionResnetB(in3f, 'in3g') # Output: 17*17
	in3h = InceptionResnetB(in3g, 'in3h') # Output: 17*17
	in3i = InceptionResnetB(in3h, 'in3i') # Output: 17*17
	in3j = InceptionResnetB(in3i, 'in3j') # Output: 17*17
	rd3 = ReductionB(in3j, 'rd3') #Output: 8*8
	# stage 4
	in4a = InceptionResnetC(rd3, 'in4a') # Output: 8*8
	in4b = InceptionResnetC(in4a, 'in4b') # Output: 8*8
	in4c = InceptionResnetC(in4b, 'in4c') # Output: 8*8
	in4d = InceptionResnetC(in4c, 'in4d') # Output: 8*8
	in4e = InceptionResnetC(in4d, 'in4e') # Output: 8*8
	# stage 5
	pool = mx.symbol.Pooling(data=in4e, kernel=(8, 8), stride=(1, 1), pool_type='avg', name='global_pool')
	drop = mx.symbol.Dropout(data=pool, p=0.2, name='dropout')
	flatten = mx.symbol.Flatten(data=drop, name='flatten')
	fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc')
	softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
	return softmax
