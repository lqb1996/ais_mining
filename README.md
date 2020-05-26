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

02-26_18:21  将block中的attention层改为mutihead-attention层，head数为7，使用大数据集时在单卡Titan RTX上batch_size为1会爆显存，因此使用mini_dataset训练,batch_size为4，每积累128个batch计算一次梯度

##clustering object
