import numpy as np
import pandas as pd
import random as rnd
from sklearn.model_selection import train_test_split

##################################################################
###all the possible sequence letters
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

def transformEL(dataset):
    dataset = dataset.reset_index(drop=True)
    peptide=dataset.Peptide
    peptide2list=peptide.tolist()
    for i in range(len(peptide)):
        if len(peptide2list[i]) < 11:
            n1 = len(peptide2list[i]) // 2
            n2 = 11 - len(peptide2list[i])
            peptide2list[i] = peptide2list[i][:n1] + 'Z'*n2 + peptide2list[i][n1:]     #将小于11个氨基酸的peptide在中间插空补齐
        else:
            peptide2list[i] = peptide2list[i][:11]

    del dataset['Peptide']
    peptides = pd.DataFrame(peptide2list,columns=['Peptide'])
    dataset.insert(0,'Peptide',peptides.pop('Peptide'))
    return dataset

def getdata_onehot(datafile):   #build testing matrix
    ### READ in test dataset
    """ Reads the test data file and extracts allele subtype,
            peptide length, and measurement type. Returns these information
            along with the peptide sequence and target values.
    """
    #train数据载入
    import os
    traindata = os.path.join("./DATA", "train_data", datafile )
    print("traindata name: ", traindata)
    df_train = pd.read_csv(traindata, header=0)
    df_train = df_train[df_train.Peptide.str.contains('X') == False]
    df_train = df_train[df_train.Peptide.str.contains('B') == False]
    df_train = df_train[df_train.Peptide.str.contains('U') == False]
    #eg.df_train = pd.read_csv('./DATA/train_data/A0202',sep="\t")
    
    #下采样
    new_df_0 = df_train.loc[df_train['BindingCategory']== 0].sample(frac = 1)
    #上采样
    df_1_list = []
    for i in range(4):
        df_1_list.append(df_train.loc[df_train['BindingCategory']== 1])
        new_df_1 = pd.concat(df_1_list)
    new_df_train = pd.concat([new_df_0,new_df_1])
    new_df_train = new_df_train.sample(frac = 1.0) #shuffle


    #X_train--补齐11mer--one_hot_matrix
    train_data=transformEL(new_df_train)
    trainMatrix = np.empty((0, 11,len(allSequences)), int)      
    for num in range(len(train_data.Peptide)):
        if num%1000 == 0:
            print(train_data.Peptide.iloc[num],num)
        trainMatrix = np.append(trainMatrix, [Pept_OneHotMap(train_data.Peptide.iloc[num])], axis=0)
    allele_name = train_data['HLA'][0]
    assert (trainMatrix.shape[0] == train_data.shape[0])

    #test数据载入
    testdata = os.path.join("./DATA", "test_data", datafile )
    df_test = pd.read_csv(testdata, header=0)
    df_test = df_test[df_test.Peptide.str.contains('X') == False]
    df_test = df_test[df_test.Peptide.str.contains('B') == False]
    df_test = df_test[df_test.Peptide.str.contains('U') == False]
    #eg.df_test = pd.read_csv('./DATA/test_data/A0202',sep="\t")

    #X_test--补齐11mer--one_hot_matrix
    test_data=transformEL(df_test)
    testMatrix = np.empty((0, 11,len(allSequences)), int)      
    for num in range(len(test_data.Peptide)):
        if num%1000 == 0:
            print(test_data.Peptide.iloc[num],num)
        testMatrix = np.append(testMatrix, [Pept_OneHotMap(test_data.Peptide.iloc[num])], axis=0)
    assert (testMatrix.shape[0] == test_data.shape[0])

    Y_train = train_data.BindingCategory
    Y_test = test_data.BindingCategory 
    #
    Y_train = Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)
    #
    trainlen = len(trainMatrix)
    ss1 = list(range(trainlen))
    rnd.shuffle(ss1)
    #
    valsize= int(1000) #Validation set size is 100 for validations dataset
    X_val1 = trainMatrix[ss1[0:valsize]]
    Y_val1 = Y_train.iloc[ss1[0:valsize]]
    X_val2 = trainMatrix[ss1[valsize:(2*valsize)]]
    Y_val2 = Y_train.iloc[ss1[valsize:(2*valsize)]]
    X_val3 = trainMatrix[ss1[(2*valsize):(3*valsize)]]
    Y_val3 = Y_train.iloc[ss1[(2*valsize):(3*valsize)]]    

    trainMatrix = np.delete(trainMatrix,ss1[0:(3*valsize)], axis=0)
    Y_train = Y_train.drop(Y_train.index[ss1[0:(3*valsize)]])
    # combine training and test datasets
    datasets={}
    datasets['X_train'] = trainMatrix
    datasets['Y_train'] = Y_train.values #traindata.BindingCategory.as_matrix()
    datasets['X_test'] = testMatrix
    datasets['Y_test'] = Y_test.values
    datasets['X_val1'] = X_val1
    datasets['Y_val1'] = Y_val1.values
    datasets['X_val2'] = X_val2
    datasets['Y_val2'] = Y_val2.values
    datasets['X_val3'] = X_val3
    datasets['Y_val3'] = Y_val3.values

    return datasets
    
#getdata_onehot function will return labes as 1 or 0 as a vector. To convert them into onehot encoded two-class format use this function
#function to convert output labels, which are 1 or 0, to two class outputs as [1,0] or [0,1]
def binary2onehot(yy):
    yy2= np.zeros((len(yy),2), dtype=int) #yy2.shape #(10547, 2)
    for num in range(len(yy)):
        if yy[num]==1:
            yy2[num,0]=1
        else:
            yy2[num,1]=1
    return yy2
#
#This function helps getting and arranging minibatch indices for DL model input at each epoch
#If that batch size and data size do not match, it randomly samples indices from the previous big pool
# and append to the remaining indices. In that case, there is one more iteration to complete, as data size seems a bit bigger.
def getIndicesofMinibatchs(featuredata, featurelabels, batchsize_, isShuffle=True):
    # by default and always in deep learning apps, shuffle should be True
    # usage: x=read_data.getIndicesofMinibatchs(featuredata=data['X_train'],
    #                featurelabels=data['Y_train'], batchsize_=40, isShuffle=True)
    datalength=len(featuredata)
    if isShuffle==True:
        tmpindx = np.arange(datalength)
        np.random.shuffle(tmpindx)
    tmp = datalength % batchsize_
    if tmp !=0:
        tmp2 = np.random.choice(tmpindx[range(datalength-tmp)], (batchsize_ - tmp ))
        tmpindx=np.append(tmpindx,tmp2)
    return tmpindx

