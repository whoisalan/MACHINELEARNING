#! /usr/bin/env python3
# -*-coding=utf-8-*-
'''
（1）图像风格迁移：给定一张普通图片和一种艺术风格图片，生成一张呈现艺术风格
    和普通图片内容的迁移图片。
（2）此次实现中使用了VGG19的卷积神经网络模型，优化过程使用了scipy.optimizer
    基于L-BFGS算法的fmin_l_bfgs_b方法
（3）每次反向优化20次写出一张图片，在代码运行过程中发现超过10次loss减少量减少
    趋于平缓，所以只写出15张图片
（4）从images文件夹中选择普通图片和风格图片，并且不同风格和内容图片中间过程生成
    的图片都在results文件夹中
 (5)由于保存权值的.h5文件较大，这里给出下载地址
    https://github.com/fchollet/deep-learning-models/
    releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
'''
import time
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
from keras import backend as K
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
#使用tensorflow环境编程
os=K.os
np=K.np
#定义目标图像长宽将长宽同时缩小为原来图形的两倍，则矩阵缩小为原来的1/4
img_rows=400
img_columns=300
#读入图片文件，以数组形式展开成三阶张量，后用numpy扩展为四阶张量
#最后使用对图片进行预处理：（1）去均值,（2）三基色RGB->BGR(3)调换维度 
def read_img(filename):
	img=load_img(filename,target_size=(img_columns,img_rows))
	img=img_to_array(img)
	img=np.expand_dims(img,axis=0)
	img=preprocess_input(img)
	return img
#写入/存储图片在results的文件夹中，将输出数组转换为三维张量，量化高度层BGR,并将BGR->RGB
#经灰度大小截断在（0,255）
def write_img(x,ordering):
	x=x.reshape((img_columns,img_rows,3))
	x[:,:,0]+=103.939
	x[:,:,1]+=116.779
	x[:,:,2]+=123.68
	x=x[:,:,::-1]
	x=np.clip(x,0,255).astype('uint8')
	result_file=('results/%s'%str(ordering+1).zfill(2))+'.png'
	if not os.path.exists('results'):
		os.mkdir('results')
	imsave(result_file,x)
	print(result_file)
#建立vgg19模型，本来卷基层+全连接层+输入层=19层，由于不是用于分类，没有用到全连接层，使用了no top模型
#no top模型权重大小为80.1M远远小于include top的权重574,7M
def vgg19_model(input_tensor):
	img_input=Input(tensor=input_tensor,shape=(300,400,3))
	#Blocks 1
	x=Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv1')(img_input)
	x=Conv2D(64,(3,3),activation='relu',padding='same',name='block1_conv2')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block1_pooling')(x)
	#Block 2
	x=Conv2D(128,(3,3),activation='relu',padding='same',name='block2_conv1')(x)
	x=Conv2D(128,(3,3),activation='relu',padding='same',name='block2_conv2')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block2_pooling')(x)
	#Block3
	x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv1')(x)
	x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv2')(x)
	x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv3')(x)
	x=Conv2D(256,(3,3),activation='relu',padding='same',name='block3_conv4')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block3_pooling')(x)
	#Block 4
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv1')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv2')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv3')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block4_conv4')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block4_pooling')(x)
	#Block 5
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv1')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv2')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv3')(x)
	x=Conv2D(512,(3,3),activation='relu',padding='same',name='block5_conv4')(x)
	x=MaxPooling2D((2,2),strides=(2,2),name='block5_pooling')(x)
	x=GlobalAveragePooling2D()(x)
	inputs=get_source_inputs(input_tensor)
	model=Model(inputs,x,name='vgg19')
	weights_path='vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
	model.load_weights(weights_path)
	#model.load_weights('mymodel_weights.h5')
	return model
#生成输入的张量,将内容，风格和迁移图像（中间量）一起输入到vgg模型中，返回三合一张量，和中间
#张量输入到VGG模型时要用到input tensor,中间计算要用到迁移图像的tensor,所以只输出这两个值
#待迁移图像初始化为一个待优化图片的占位符，初始输入为随机噪声图像，然后是一直优化的图像
def create_tensor(content_path,style_path):
	content_tensor=K.variable(read_img(content_path))
	style_tensor=K.variable(read_img(style_path))
	transfer_tensor=K.placeholder((1,img_columns,img_rows,3))
	input_tensor=K.concatenate([content_tensor,style_tensor,transfer_tensor],axis=0)
	return input_tensor,transfer_tensor
#设置Gram_matrix矩阵的计算图，输入为某一层的representation,Gram 矩阵表示向量组的相关性，用于求解
#迁移图像关于风格图像的loss
def gram_matrix(x):
	features=K.batch_flatten(K.permute_dimensions(x,(2,0,1)))
	gram=K.dot(features,K.transpose(features))
	return gram
#计算风格的loss,以风格图像和迁移图像的representation为输入，分别计算gram矩阵，再求解两个Gram矩阵的
#二范数，除以归一化值
def style_loss(style_img_representation,transfer_img_representation):
	style=style_img_representation
	transfer=transfer_img_representation
	A=gram_matrix(style)
	G=gram_matrix(transfer)
	channels=3
	size=img_rows*img_columns
	loss=K.sum(K.square(A-G))/(4.*(channels**2)*(size**2))
	return loss
#计算内容loss,输入为内容和迁移图片的presentation，输出为其reprensentation差的二范数
def content_loss(content_img_representation,transfer_img_representation):
	content=content_img_representation
	transfer=transfer_img_representation
	loss=K.sum(K.square(transfer-content))
	return loss		 
#变量loss,一段迷一样的表达式×-×，施加全局差正则表达式，全局差正则用于使生成的图片更加平滑自然
def total_variation_loss(x):
	a=K.square(x[:,:img_columns-1,:img_rows-1,:]-x[:,1:,:img_rows-1,:])
	b=K.square(x[:,:img_columns-1,:img_rows-1,:]-x[:,:img_columns-1,1:,:])
	loss=K.sum(K.pow(a+b,1.25))
	return loss
#建立层名称到层输出张量映射的dict,便于取得各层输出feature map,分别求解style loss和content loss
#风格loss和内容loss按照10：1结合为全变量差约束
def total_loss(model,loss_weights,transfer_tensor):
	loss=K.variable(0.)
	layer_features_dict=dict([(layer.name,layer.output) for layer in model.layers])
	layer_features=layer_features_dict['block4_conv2']
	content_img_features=layer_features[0,:,:,:]
	transfer_img_features=layer_features[2,:,:,:]
	loss+=loss_weights['content']*content_loss(content_img_features,transfer_img_features)
	feature_layers=['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
	for layer_name in feature_layers:
		layer_features=layer_features_dict[layer_name]
		style_img_features=layer_features[1,:,:,:]
		transfer_img_features=layer_features[2,:,:,:]
		loss+=(loss_weights['style']/len(feature_layers))*(style_loss(style_img_features,transfer_img_features))
	loss+=loss_weights['total']*total_variation_loss(transfer_tensor)
	return loss
#通过K.gradient获取反向梯度，同时得到梯度和损失，
def create_outputs(total_loss,transfer_tensor):
	gradients=K.gradients(total_loss,transfer_tensor)
	outputs=[total_loss]
	if isinstance(gradients,(list,tuple)):
		print('list/tuple')
		outputs+=gradients
	else:
		outputs.append(gradients)
	return outputs
#计算输入图像的关于损失函数的梯度值和对应损失值
def eval_loss_and_grads(x):
	x=x.reshape((1,img_columns,img_rows,3))
	outs=outputs_func([x])
	loss_value=outs[0]
	if len(outs[1:])==1:
		grads_value=outs[1].flatten().astype('float64')
	else:
		grads_value=np.array(outs[1:]).flatten().astype('float64')
	return loss_value,grads_value
#获取评价程序，将获取计算loss和gradients的函数
class Evaluator(object):
	def __init__(self):
		self.loss_value=None
		self.grads_value=None
	def loss(self,x):
		loss_value,grads_value= eval_loss_and_grads(x)
		self.loss_value=loss_value
		self.grads_value=grads_value
		return self.loss_value
	def grads(self,x):
		grads_value=np.copy(self.grads_value)
		self.loss_value=None
		self.grads_value=None
		return grads_value
#main函数
if __name__=='__main__':
	print('')
	print('Welcom!')
	#输入图片路径
	path={'content':'images/k.jpg','style':'images/StarryNight.jpg'}
	#path={'content':'images/Taipei101.jpg','style':'images/guernica.jpg'}
	input_tensor,transfer_tensor=create_tensor(path['content'],path['style'])
	#用来计算总loss的系数
	#loss_weights={'style':1.0,'content':0.025,'total':1.0}
	loss_weights={'style':1.0,'content':0.01,'total':1.0}
	model=vgg19_model(input_tensor)
	#生成总的反向特征缺失
	total_loss=total_loss(model,loss_weights,transfer_tensor)
	#生成正向输出
	outputs=create_outputs(total_loss,transfer_tensor)
	#获取计算图(反向输入图)
	outputs_func=K.function([transfer_tensor],outputs)
	#生成处理器
	evaluator=Evaluator()
	#生成噪声
	x=np.random.uniform(0,225,(1,img_columns,img_rows,3))-128
	#迭代训练15次
	for ordering in range(15):
		print('Start:',ordering+1)
		start_time=time.time()
		x,min_val,info=fmin_l_bfgs_b(evaluator.loss,x.flatten(),fprime=evaluator.grads,maxfun=20)
		print('Current_Loss:',min_val)
		img=np.copy(x)
		write_img(img,ordering)
		end_time=time.time()
		print('Used %ds'%(end_time-start_time))
	model.save_weights('mymodel_weights.h5')