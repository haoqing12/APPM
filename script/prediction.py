import tensorflow as tf
import numpy as np
import pandas as pd
import math, os, time, sys, datetime
from datetime import timedelta
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import stats
# python script/prediction.py  A0101

allSequences = 'ACEDGFIHKMLNQPSRTWVYZ'
# Establish a mapping from letters to integers
char2int = dict((c, i) for i, c in enumerate(allSequences))


##################################################################
def Pept_OneHotMap(peptideSeq):
    """ maps amino acid into its numerical index
    USAGE
    Pept_OneHotMap('A')
    array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """
    # integer encode input data
    integer_encoded=[char2int[char] for char in peptideSeq]
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
    	letter = [0 for _ in range(len(allSequences))]
    	letter[value] = 1
    	onehot_encoded.append(letter)
    return np.asarray(onehot_encoded)
###################################################################

def transformEL(dataset):
    dataset = dataset.reset_index(drop=True)
    peptide=dataset.Peptide
    peptide2list=peptide.tolist()

    for i in range(len(peptide)):
        if len(peptide2list[i]) < 11:
            n1 = len(peptide2list[i]) // 2
            n2 = 11 - len(peptide2list[i])
            peptide2list[i] = peptide2list[i][:n1] + 'Z'*n2 + peptide2list[i][n1:]
        else:
            peptide2list[i] = peptide2list[i][:11]

    del dataset['Peptide']
    peptides = pd.DataFrame(peptide2list,columns=['Peptide'])
    dataset.insert(0,'Peptide',peptides.pop('Peptide'))
    return dataset
#####################################################################

def getdata_onehot(predictdatafile):
    ### READ in test dataset
    """ Reads the test data file and extracts allele subtype,
            peptide length, and measurement type. Returns these information
            along with the peptide sequence and target values.
    """
    print("Test peptide name: ", predictdatafile)
    import os
    predict_set = os.path.join("./DATA", "predict_data", predictdatafile )
    print("test_set name: ", predict_set)          #sys.argv[1]:A0101
    predict_data = pd.read_csv(predict_set, header=0)
    predict_data = predict_data[predict_data.Peptide.str.contains('X') == False]
    predict_data = predict_data[predict_data.Peptide.str.contains('B') == False]
    predict_data = predict_data[predict_data.Peptide.str.contains('U') == False]
    #predict_data = pd.read_csv('./DATA/predict_data/A0101',sep="\t")
    
    predictdata = pd.DataFrame()
    predictdata=transformEL(predict_data)
    predictMatrix = np.empty((0, 11,len(allSequences)), int)      #测试集
    for num in range(len(predictdata.Peptide)):
        if num%1000 == 0:
            print(predictdata.Peptide.iloc[num],num)
        predictMatrix = np.append(predictMatrix, [Pept_OneHotMap(predictdata.Peptide.iloc[num])], axis=0)


    datasets={}
    datasets['X_predict'] = predictMatrix   
    return datasets
###################################################################
def conv2d_layer(input_data, num_input_channels, num_filters, filter_shape,strides_, name):   #(x, W, b, strides_=[2,2]):
    # Conv2D wrapper, with bias and relu activation
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    # initialise weights and bias for the filter
    weights = tf.Variable(tf.random.truncated_normal(conv_filt_shape, stddev=0.01), name=name+'_W') #, seed=myseed[2]
    bias = tf.Variable(tf.random.truncated_normal([num_filters]), name=name+'_b')  #, seed=myseed[3],
    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input=input_data, filter=weights, strides=[1, strides_[0], strides_[1], 1], padding='SAME')  #see below, Exp1, for the explanations of conv2d if needed.
    # add the bias
    out_layer = tf.nn.bias_add(out_layer, bias)
    # apply a ReLU non-linear activation (or change as you like)
    return tf.nn.leaky_relu(features=out_layer, alpha=0.2)


####################################################################

numofinput_channels = 1 # 1 data input per feature
numofclasses=2  # data labels are binary.
#for dropout probability
prob_ = tf.compat.v1.placeholder( dtype=tf.float32, shape=() )
keep_prob_rate=0.5 #0.4
nnodes_f1= 100
######### Design the DL model   #########
# Tensor inputs for 4-D: [Batch Size, Height, Width, Channel]
def DL_model(inputData):
    # add a custom 2D Convolution Layer
    #output1a has different filter shape
    nfilters1a=128 #can be 512
    output1a= conv2d_layer(input_data=inputData , num_input_channels=numofinput_channels,
                num_filters=nfilters1a, filter_shape=[2,2],strides_=[1,1], name='CNN2d_1_a')
    print("CNN2d_1_a output shape: ", output1a.get_shape())
    nfilters2a=128
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters1a,
                num_filters=nfilters2a, filter_shape=[2,2],strides_=[2,2], name='CNN2d_2_a')
    print("CNN2d_2_a output shape: ", output1a.get_shape())
    #**
    nfilters3a=256
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters2a,
                num_filters=nfilters3a, filter_shape=[2,2],strides_=[2,2], name='CNN2d_3_a')
    print("CNN2d_3_a output shape: ", output1a.get_shape())
    nfilters4a=256
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters3a,
                num_filters=nfilters4a, filter_shape=[2,2],strides_=[2,2], name='CNN2d_4_a')
    print("CNN2d_4_a output shape: ", output1a.get_shape())
    nfilters5a=256
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters4a,
                num_filters=nfilters5a, filter_shape=[2,2],strides_=[2,2], name='CNN2d_5_a')
    print("CNN2d_5_a output shape: ", output1a.get_shape())
    nfilters6a=512
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters5a,
                num_filters=nfilters6a, filter_shape=[1,2],strides_=[1,2], name='CNN2d_6_a')
    print("CNN2d_6_a output shape: ", output1a.get_shape())
    nfilters7a=512
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters6a,
                num_filters=nfilters7a, filter_shape=[1,1],strides_=[1,1], name='CNN2d_7_a')
    print("CNN2d_7_a output shape: ", output1a.get_shape())
    nfilters8a=256
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters7a,
                num_filters=nfilters8a, filter_shape=[1,1],strides_=[1,1], name='CNN2d_8_a')
    print("CNN2d_8_a output shape: ", output1a.get_shape())
    '''
    nfilters9a=256
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters8a,
                num_filters=nfilters9a, filter_shape=[1,1],strides_=[1,1], name='CNN2d_9_a')
    print("CNN2d_9_a output shape: ", output1a.get_shape())
    '''
    #**
    lastfiltersize_a=nfilters8a
    out1a_h = output1a.get_shape().as_list()[1]; out1a_w  = output1a.get_shape().as_list()[2]
    output1a_reshape = tf.reshape(output1a, [-1, out1a_h*out1a_w*lastfiltersize_a])
    print("CNN2d_1_a output reshaped: ", output1a_reshape.get_shape())
    #######layer B: output1b has different filter shape
    nfilters1b=128
    output1b= conv2d_layer(input_data=inputData , num_input_channels=numofinput_channels,
                num_filters=nfilters1b, filter_shape=[1,2],strides_=[1,1], name='CNN2d_1_b')
    print("CNN2d_1_b output shape: ", output1b.get_shape())
    nfilters2b=128
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters1b,
                num_filters=nfilters2b, filter_shape=[2,2],strides_=[2,2], name='CNN2d_2_b')
    print("CNN2d_2_b output shape: ", output1b.get_shape())
    #**
    nfilters3b=256
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters2b,
                num_filters=nfilters3b, filter_shape=[2,2],strides_=[2,2], name='CNN2d_3_b')
    print("CNN2d_3_b output shape: ", output1b.get_shape())
    nfilters4b=256
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters3b,
                num_filters=nfilters4b, filter_shape=[2,2],strides_=[2,2], name='CNN2d_4_b')
    print("CNN2d_4_b output shape: ", output1b.get_shape())
    nfilters5b=256
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters4b,
                num_filters=nfilters5b, filter_shape=[2,2],strides_=[2,2], name='CNN2d_5_b')
    print("CNN2d_5_b output shape: ", output1b.get_shape())
    nfilters6b=512
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters5b,
                num_filters=nfilters6b, filter_shape=[1,2],strides_=[1,2], name='CNN2d_6_b')
    print("CNN2d_6_b output shape: ", output1b.get_shape())
    nfilters7b=512
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters6b,
                num_filters=nfilters7b, filter_shape=[1,1],strides_=[1,1], name='CNN2d_7_b')
    print("CNN2d_7_b output shape: ", output1b.get_shape())
    nfilters8b=256
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters7b,
                num_filters=nfilters8b, filter_shape=[1,1],strides_=[1,1], name='CNN2d_8_b')
    print("CNN2d_8_b output shape: ", output1b.get_shape())
    '''
    nfilters9b=128
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters8b,
                num_filters=nfilters9b, filter_shape=[1,1],strides_=[1,1], name='CNN2d_9_b')
    print("CNN2d_9_b output shape: ", output1b.get_shape())
    '''
    #**
    lastfiltersize_b=nfilters8b
    out1b_h = output1b.get_shape().as_list()[1]; out1b_w  = output1b.get_shape().as_list()[2]
    output1b_reshape = tf.reshape(output1b, [-1, out1b_h*out1b_w*lastfiltersize_b])
    print("CNN2d_1_b output reshaped: ", output1b_reshape.get_shape())
    ########
    #######layer C: output1c has different filter shape
    nfilters1c=128
    output1c= conv2d_layer(input_data=inputData , num_input_channels=numofinput_channels,
                num_filters=nfilters1c, filter_shape=[2,1],strides_=[1,1], name='CNN2d_1_c')
    print("CNN2d_1_c output shape: ", output1c.get_shape())
    nfilters2c=128
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters1c,
                num_filters=nfilters2c, filter_shape=[2,2],strides_=[2,2], name='CNN2d_2_c')
    print("CNN2d_2_c output shape: ", output1c.get_shape())
    #**
    nfilters3c=256
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters2c,
                num_filters=nfilters3c, filter_shape=[2,2],strides_=[2,2], name='CNN2d_3_c')
    print("CNN2d_3_c output shape: ", output1c.get_shape())
    nfilters4c=256
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters3c,
                num_filters=nfilters4c, filter_shape=[2,2],strides_=[2,2], name='CNN2d_4_c')
    print("CNN2d_4_c output shape: ", output1c.get_shape())
    nfilters5c=256
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters4c,
                num_filters=nfilters5c, filter_shape=[2,2],strides_=[2,2], name='CNN2d_5_c')
    print("CNN2d_5_c output shape: ", output1c.get_shape())
    nfilters6c=512
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters5c,
                num_filters=nfilters6c, filter_shape=[1,2],strides_=[1,2], name='CNN2d_6_c')
    print("CNN2d_6_c output shape: ", output1c.get_shape())
    nfilters7c=512
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters6c,
                num_filters=nfilters7c, filter_shape=[1,1],strides_=[1,1], name='CNN2d_7_c')
    print("CNN2d_7_c output shape: ", output1c.get_shape())
    nfilters8c=256
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters7c,
                num_filters=nfilters8c, filter_shape=[1,1],strides_=[1,1], name='CNN2d_8_c')
    print("CNN2d_8_c output shape: ", output1c.get_shape())
    '''
    nfilters9c=256
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters8c,
                num_filters=nfilters9c, filter_shape=[1,1],strides_=[1,1], name='CNN2d_9_c')
    print("CNN2d_9_c output shape: ", output1c.get_shape())
    '''
    #**
    lastfiltersize_c=nfilters8c
    out1c_h = output1c.get_shape().as_list()[1]; out1c_w  = output1c.get_shape().as_list()[2]
    output1c_reshape = tf.reshape(output1c, [-1, out1c_h*out1c_w*lastfiltersize_c])
    print("CNN2d_1_c output reshaped: ", output1c_reshape.get_shape())
    ########

    #COMBINE THREE PARALLEL CONV CONNECTIONS OF DIFFERENT FILTER SIZES HERE
    flattened = tf.concat([output1a_reshape, output1b_reshape,output1c_reshape], axis=1) #combined the two praralel filters
    out_height = flattened.get_shape().as_list()[1]
    # Fully connected layer
    # Reshape conv2 output1 to fit fully connected layer input
    #flattened = tf.reshape(output1, [-1, out_height * out_width * nfilters2])
    print("flattened layer output shape: ", flattened.get_shape())
    # setup some weights and bias values for this layer, then activate with ReLU
    #nnodes_f1 is set at the begining
    W_f1 = tf.Variable(tf.random.truncated_normal([out_height, nnodes_f1], stddev=0.01),  name='W_f1') #, seed=myseed[4]
    B_f1 = tf.Variable(tf.random.truncated_normal([nnodes_f1], stddev=0.01),  name='B_f1') #, seed=myseed[5]
    #
    dense_layer1 = tf.add(tf.matmul(flattened, W_f1), B_f1)
    dense_layer1 = tf.nn.leaky_relu(features=dense_layer1, alpha=0.2)
    print("dense_layer1 output shape: ", dense_layer1.get_shape())
    #
    # Apply Dropout
    dense_layer1 = tf.nn.dropout(x=dense_layer1, rate=1-prob_) #Dropout process

    # another layer for the final output
    wd2 = tf.Variable(tf.random.truncated_normal([nnodes_f1, numofclasses], stddev=0.01), name='wd2') #, seed=myseed[6]
    bd2 = tf.Variable(tf.random.truncated_normal([numofclasses], stddev=0.01),  name='bd2') #, seed=myseed[7]
    final_layer = tf.add(tf.matmul(dense_layer1, wd2), bd2) #class prediction
    print("final_layer output shape: ", final_layer.get_shape())
    return final_layer
###################################################################
#将[0,1]变回[0],将[1,0]变回[1]
def onehot2binary(yy):
    yy2= np.zeros((len(yy),1),dtype=int)
    for num in range(len(yy)):
        if yy[num][0] > yy[num][1]:
            yy2[num] = int(1)
        else:
            yy2[num] = int(0)
    return yy2
####################################################################

#载入数据
print("Tensorflow version " + tf.__version__)
# python script/prediction.py  A0101 
predictset = sys.argv[1]   #A0101
print("Test set is ", predictset)
data = getdata_onehot(predictdatafile=predictset)


input_height = data['X_predict'][0].shape[0] #11, depends onpeptide length
input_width = data['X_predict'][0].shape[1]  #21 , comes from size of unique peptide sequence letters
# Tensor graph input is 4-D: [Batch Size, Height, Width, Channel]
X = tf.compat.v1.placeholder(tf.float32, shape=[None, input_height, input_width])
# dynamically reshape the input
X_shaped = tf.reshape(X, [-1, input_height, input_width, 1])
#None: 'number of' (#) input is dynamic, not decied yet. numofinput_channels is 1 input channel
# now declare the output data placeholder - 2 digits
####################################################################

#数据导入神经网络
save_dir = 'checkpoints/' + predictset + '/'
save_path = os.path.join(save_dir, 'best_validation')
save_thelast_path = os.path.join(save_dir, 'last_weigths')
logits = DL_model(inputData=X_shaped)
prediction = tf.nn.softmax(logits)


#运用训练好的模型预测
with tf.compat.v1.Session() as sess:
    #Restore the last saved best model model
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess=sess, save_path=save_path)
    graph = tf.compat.v1.get_default_graph()
    predictions = sess.run(prediction, feed_dict={X: data['X_predict'], prob_: 1.0})
    prediction_val = onehot2binary(predictions)

predict_set = os.path.join("./DATA", "predict_data", predictset )         #sys.argv[1]:A0101
dataset = pd.read_csv(predict_set, header=0)


prediction_ = pd.DataFrame(prediction_val,columns=['predictresult'])
dataset.insert(2,'predictresult',prediction_.pop('predictresult'))

predict_dir = 'predictresults/' + predictset + '/'
if not os.path.exists(predict_dir):
    os.makedirs(predict_dir)

dataset.to_csv(predict_dir+predictset+'.csv',index=False)
