#Ais Data Mining
##predict object
loss:MSE
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
##clustering object