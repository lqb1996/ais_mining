#Ais Data Mining
##predict task
loss为MSE loss
just lstm
loss:0.3
lstm+dropout
loss:0.05
lstm+dropout+mixed_loss(panished more on offset)+cancle_batch_first
loss:0.03
just ground_truth
loss:0.03
add ELU activation function(predict_offset)
loss:0.003
get_some_use
predict_offset add more loss about offset and add attention layer
loss:0.014
add log square loss and add two lstm structure and add layer normalization
loss:0.0006
remove log square, add two lstm structure and add layer normalization
loss:0.0007
have a state of art stucture
a larger model with two self attention and a resnet,loss is same as
loss:
take a modle like transformer
loss:0.01,just so so
改进transfomer，增加全局残差结构(效果不错)
loss:0.0002
设置num_block=2和num_block=6均出现梯度爆炸问题
其中num_block=2梯度尤为明显，且训练过程中不收敛
设置max_norm=12的L2梯度裁剪和学习率衰减
loss在0.003,4,50之间跳动，全局没有下降
设置梯度累加 
loss依然跳动，影响不大
设置固定值的梯度裁剪,影响不大
去掉平方项loss，每32个batch更新一次梯度（模拟batchsize=16*32）
新增条件推理后的数据，根据速度和方向计算目标在各方向的偏移量
loss为mse和实际loss的和
仅计算偏移量mse损失时，总是会将预测点归一到0

输入数据的维度改为7维:cog,rot,trueHeading,sog,预测点与当前点的时间差,根据方向和速度预测的经度偏移量,根据方向和速度预测的纬度偏移量
05-23_00:07 将lambda值设为100,loss收敛至30左右,并有一定效果,MSE loss为0.3左右，block num为2，但是有很多可以改进的地方

05-24_23:15 尝试使用传统的lstm,2层的双向lstm，在重新做好数据治理的基础上，loss为offset和实际loss的平方和

05-24_23:39 尝试使用传统的lstm+attention，在重新做好数据治理的基础上，loss为offset和实际loss的平方和

05-25_20:40 将loss设为offset和实际loss的平方和,block num设为7

###对比方案1
02-26_18:21 将block中的attention层改为mutihead-attention层，head数为7，使用大数据集时在单卡Titan RTX上batch_size为1会爆显存，因此使用mini_dataset训练,batch_size为4，每积累128个batch计算一次梯度
loss最低能降到0.0002，但是视觉上效果很差,有一定预测能力,且为阶段最优的方案，

05-27_14:10 使用经典bert结构外接lstm最后两层fc输出结果，仅对lstm设置dropout=0.4

05-27_16:55 将7维的结构输入lstm后经过bert结构再接入一个lstm+fc输出结果

05-27_23:54 沿用02-26_18:21的方案，为trans_block加上mask测试结果，调整lr衰减参数，使lr衰减更慢
效果有提升,loss收敛在0.48
###对比方案2
05-28_00:08 为了验证全局LN层的有效性，使用LSTMTrans4PRE模型，沿用02-26_18:21的方案，注意做test时需要调整trans_block的hidenlayer，并且去掉了两个全局的LN层
效果有所提升,train_loss收敛在0.3,且为阶段最优的方案

05-28_10:51 在05-28_00:08基础上，注意做test时需要调整trans_block的hidenlayer*7，block中的res结构改为先过LN再res
效果没有提升，与原模型效果相近但比原模型效果差一些

05-28_15:24 在05-28_00:08基础上，block中的res结构改为先过LN再res，不调整trans_block的hidenlayer，hidenlayer*2
先LN后再加残差结构会使初开始的loss非常高，epoch=30是loss为2.3，预计模型效果没有原方案好
###对比方案3
05-28_17:48 在05-28_00:08方案中加入了4个维度分别为经纬度和偏移量的卡尔曼滤波结果
效果惊人...收敛极快...

05-28_18:57 在05-28_00:08方案中加入了8个维度分别为经纬度和偏移量在R值分别取1*2和0.001*2时的卡尔曼滤波结果,num_block=16
多头注意力改为15个,效果不稳定，待优化

05-29_11:22 增加输入维度size=32，将所有信息都输入网络，减少hiddenlayer=2*32，减少多头注意力改为32个
效果惊人，但随着训练继续loss反而增加并且波动频繁，loss最低时为0.0093

###最优方案(lstm_large.KalmanTrans4PRE_large)
05-30_00:23 在05-29_11:22基础上去除了几个影响不大的输入特征，再次进行训练,input_size=24,muti_head_num=24
效果比原方案好些,loss能够在较低位置收敛,目前最低时为0.0063
epoch=1177时,loss=0.0063,val_loss=0.0002
但是模型只学习到了轨迹的趋势，即使loss包含了MSE的惩罚，整体的轨迹和真实轨迹仍然偏差很大

将输入作为全局残差放入fc的前一层

05-30_18:01 使用05-30_00:23方案中的维度沿用bert后加lstm的思路，加入卡尔曼滤波特征后是否有所改进
效果依然很差，有待优化

测试单纯使用卡尔曼滤波作为预测时的效果

05-31_00:48 验证一下05-30_00:23中mask的效果，去掉attention中的mask
起初的loss和原方案差不多，但下降的很慢且收敛的比较高

基于05-30_00:23的方案，由于该方案能很好的学习到轨迹的趋势但是整体轨迹会与原轨迹有偏移
由于原方案是将lstm的特征进行残差，于是改为将输入的上采样直接输入各层做残差
loss很大，下降缓慢

基于05-30_00:23的方案，仅将输入的上采样与lstm的输出和encoder的输出进行cat作为全局残差输入

##卡尔曼滤波(Kalman Filter)
baseline
