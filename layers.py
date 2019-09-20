from graphcnn.helper import * #changed
import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib.layers.python.layers import utils

# Refer to Such et al for details
         
def make_variable(name, shape, initializer=tf.truncated_normal_initializer(), regularizer=None):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=dtype)
    return var
    
def make_bias_variable(name, shape):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1), dtype=dtype)
    return var

def make_variable_with_weight_decay(name, shape, stddev=0.01, wd=0.0005):
    dtype = tf.float32
    regularizer = None
    if wd is not None and wd > 1e-7:
        def regularizer(var):
            return tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    var = make_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev), regularizer=regularizer)
    return var
    
def make_bn(input, phase, axis=-1, epsilon=0.001, mask=None, num_updates=None, name=None):
    default_decay = GraphCNNGlobal.BN_DECAY
    with tf.variable_scope(name, default_name='BatchNorm') as scope:
        input_size = input.get_shape()[axis].value
        if axis == -1:
            axis = len(input.get_shape())-1
        axis_arr = [i for i in range(len(input.get_shape())) if i != axis]
        if mask == None:
            batch_mean, batch_var = tf.nn.moments(input, axis_arr)
        else:
            # weighted moment uses label weights (implemented in this work)
            batch_mean, batch_var = tf.nn.weighted_moments(input, axis_arr, mask)
        gamma = make_variable('gamma', input_size, initializer=tf.constant_initializer(1))
        beta = make_bias_variable('bias', input_size)
        ema = tf.train.ExponentialMovingAverage(decay=default_decay, num_updates=num_updates)
        
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(tf.cast(tf.equal(phase,1),dtype=tf.bool), mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        
        return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
      
def batch_mat_mult(A, B):
    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, [-1, A_shape[-1]])
    
    # So the Tensor has known dimensions
    if B.get_shape()[1] == None:
        axis_2 = -1
    else:
        axis_2 = B.get_shape()[1]
    result = tf.matmul(A_reshape, B)
    result = tf.reshape(result, tf.stack([A_shape[0], A_shape[1], axis_2]))
    return result

def batch_mat_mult1(A, B):
    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, [-1, A_shape[-1]])
    
    # So the Tensor has known dimensions
    if B.get_shape()[1] == None:
        axis_2 = -1
    else:
        axis_2 = B.get_shape()[1]
    result = tf.matmul(A_reshape, B)
    result = tf.reshape(result, tf.stack([A_shape[0], A.shape[1], axis_2]))
    return result
def batch_mat_mult2(A, B):
    B_shape = tf.shape(B)
    B_reshape = tf.reshape(B, [-1, B_shape[-1]])
    
    # So the Tensor has known dimensions
    
    result = tf.matmul(A,B_reshape)
    result = tf.reshape(result, tf.stack([B_shape[0], A.shape[1], B_shape[-1]]))
    return result   
def make_softmax_layer(V, axis=1, name=None):
    with tf.variable_scope(name, default_name='Softmax') as scope:
        max_value = tf.reduce_max(V, axis=axis, keep_dims=True)
        exp = tf.exp(tf.subtract(V, max_value))
        prob = tf.div(exp, tf.reduce_sum(exp, axis=axis, keep_dims=True))
        return prob
    
def make_graphcnn_layer(V, A, no_filters, flag=False, name=None):
    with tf.variable_scope(name, default_name='Graph-CNN') as scope:
        # if flag==True:
            # no_A = A.get_shape()[1].value
            # no_features = V.get_shape()[1].value
        no_A = A.get_shape()[2].value
        no_features = V.get_shape()[2].value
        # print_ext("Shape of V:",V.get_shape())
        # print_ext("no of nodes in output:",V.get_shape()[1].value)
        # print_ext("no of features in output:",no_filters)
        W = make_variable_with_weight_decay('weights', [no_features*no_A, no_filters], stddev=math.sqrt(1.0/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        W_I = make_variable_with_weight_decay('weights_I', [no_features, no_filters], stddev=math.sqrt(GraphCNNGlobal.GRAPHCNN_I_FACTOR/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        b = make_bias_variable('bias', [no_filters])

        A_shape = A.get_shape()
        A_reshape = tf.reshape(A, tf.stack([-1, A_shape[1]*no_A, A_shape[1]]))
        n = tf.matmul(A_reshape, V)
        n = tf.reshape(n, [-1, tf.shape(A)[1], no_A*no_features])
        result = batch_mat_mult(n, W) + batch_mat_mult(V, W_I) + b
        return result

def make_graph_embed_pooling(V, A, no_vertices=1, flag=False, mask=None, name=None):
    with tf.variable_scope(name, default_name='GraphEmbedPooling') as scope:
        # print_ext("Shape of V:",V.get_shape())
        # print_ext("no of nodes in output:",no_vertices)
        # print_ext("no of features in output:",V.get_shape()[2].value)
        
        factors = make_embedding_layer(V, no_vertices, name='Factors')
        
        #if mask is not None:
        #   factors = tf.multiply(factors, tf.to_float(mask))
        factors = make_softmax_layer(factors)
        
        result = tf.matmul(factors, V, transpose_a=True)
        
        
        if no_vertices == 1:
            # if flag==True:
                
            no_features = V.get_shape()[2].value
            return tf.reshape(result, [-1, no_features]), A
        
        
        
        result_A = tf.reshape(A, (tf.shape(A)[0], -1, tf.shape(A)[-1]))
        result_A = tf.matmul(result_A, factors)
        result_A = tf.reshape(result_A, (tf.shape(A)[0], tf.shape(A)[-1], -1))
        result_A = tf.matmul(factors, result_A, transpose_a=True)
        result_A = tf.reshape(result_A, (tf.shape(A)[0], no_vertices, A.get_shape()[2].value, no_vertices))
        
        return result, result_A
    
def make_embedding_layer(V, no_filters, name=None):
    with tf.variable_scope(name, default_name='Embed') as scope:
        # print_ext("Shape of V:",V.get_shape())
        # print_ext("no of nodes in output:",V.get_shape()[1].value)
        # print_ext("no of features in output:",no_filters)
        
        no_features = V.get_shape()[-1].value
        W = make_variable_with_weight_decay('weights', [no_features, no_filters], stddev=1.0/math.sqrt(no_features))
        b = make_bias_variable('bias', [no_filters])
        V_reshape = tf.reshape(V, (-1, no_features))
        # print_ext("shape of V is:",V_reshape.get_shape())
        s = tf.slice(tf.shape(V), [0], [len(V.get_shape())-1])
        s = tf.concat([s, tf.stack([no_filters])], 0)
        # print_ext("value of no_filters is:",no_filters)
        # print_ext("value of s is:",np.array(s))
        result = tf.reshape(tf.matmul(V_reshape, W) + b, s)
        return result
'''
def graph_attention(V, A, F_, attn_heads ): 
     
    # Parameters:
    # F = input cardinality of features
    # F_ = output cardinality of features
    # attn_heads = (K in the paper) no. of independent computations
    # Assuming V is output feature of previous layer
    # A is adjacency matrix
     
    with tf.variable_scope(None, default_name='attention') as scope:
        h = []
        for k in range(attn_heads):
            no_features = V.get_shape()[-1].value
            W = make_variable_with_weight_decay('weights-{}'.format(str(k)), [F_, no_features], stddev=1.0/math.sqrt(no_features)) #debug shape
            #print('sqeeze',tf.shape(tf.squeeze(V,axis = 0)))
            # V_temp = tf.slice(V, [0,0,0], [1,V.shape[1],V.shape[2]])
            # V_temp = tf.matmul(V_temp,W)
            V_ = batch_mat_mult1(V, W )   #V_ is a vector of shape N*F_, similar to W*h in the paper. N is no. of nodes
            print('shape V_',V_.shape)

            #PA = tf.get_variable("PA-{}".format(str(k)), shape = (A.shape[1], A.shape[3]) , initializer=tf.constant_initializer(0.1), dtype=tf.float32, trainable = False)
            #buff = tf.get_variable("buff-{}".format(str(k)), shape = (A.shape[0], A.shape[0], 2*F_) , initializer=tf.constant_initializer(0.1), dtype=dtype, trainable = False)
            #e = tf.get_variable("e-{}".format(str(k)), shape = (A.shape[0], A.shape[0], 2*F_) , initializer=tf.constant_initializer(0.1), dtype=dtype, trainable = False)
            dtype = tf.float32
            #alpha = tf.get_variable("alpha-{}".format(str(k)), shape = (A.shape[1], A.shape[1]) , initializer=tf.constant_initializer(0.1), dtype=tf.float32, trainable = False)
            PA = [] #np.empty((A.shape[0]*A.shape[0]*2*F_)) # , initializer=tf.constant_initializer(0.1), dtype=dtype, trainable = False)
            alpha =[]
            #sum_PA = []
            
            
            a = make_variable_with_weight_decay('a-{}'.format(str(k)), [2*F_,1], stddev=1.0/math.sqrt(no_features)) #debug shape
            #print(A.shape)
            for i in range(A.shape[1]) :
                
                for j in range(A.shape[3]) :
                	#print('j',j)
                    #P[i][j] = tf.concat([V_[i],V_[j]], axis = 0 )  #debug shape #may have to reshape
                    #P(i,j) will contain the concatenated vectors W*h(i) and W*h(j)
                    #buff[i][j] = tf.tensordot(tf.concat([V_[i],V_[j]], axis = 0 ),a)    #forward propagation before leakyReLU
                    #e[i][j] = tf.nn.leaky_relu(tf.tensordot(tf.concat([V_[i],V_[j]], axis = 0 ),a), alpha = 0.2 )    #As in the paper
                    buff = tf.concat([V_[:,i],V_[:,j]], axis = 1 )
                    # b_shape = tf.shape(buff) #Roundabout way to extract ? shape (As in batch_mat_mult function)
                    buff = tf.expand_dims(buff,axis=1)
                    # print("Buff_Shape",buff.shape)
                    PA.append(tf.math.scalar_mul(tf.reshape(A,[A.shape[1],A.shape[3]])[i][j], tf.nn.leaky_relu(batch_mat_mult1(buff,a), alpha = 0.2 )))
                    # print('shape', (batch_mat_mult1(buff,a).shape))
                    #PA[i][j] entry is zero if vertices i and j are not neighbours.
                    #alpha[i][j] = make_softmax_layer(e[i][j], axis=1) #debug
                    #TODO - for each column in e, alpha[i][j] is
                    # exp(e[i][j])/(sum((exp(e[i][k]))) if A[i][k] is 1
            PA = np.array(PA)
            #PA = tf.stack(PA)
            #print('shape',PA.shape)
            #print('PA_value',PA[1])
            PA = np.reshape(PA,(A.shape[1], A.shape[3]))
            #PA[0][0] = tf.constant(0)
            #print('PA[0][0]',PA[0][0])
            for i in range(A.shape[1]) :
                div = 0
                print('i',i)
                for j in range(A.shape[3]) :
                    if (PA[i][j] != 0):
                    	#print('true')
                    	div = div + tf.exp(PA[i][j])
                    
                for j in range(A.shape[3]) :
                    alpha.append(tf.div(PA[i][j], div))
                    # print('paij',PA[i][j].shape)

            #alpha = np.array(alpha)
            alpha = tf.stack(alpha)
            #print('alpha',np.shape(alpha))
            #print('value',alpha[1])
            alpha = tf.reshape(alpha,(A.shape[1], A.shape[3]))
            # print(tf.shape(alpha))
            #print('value1',alpha[1][1])
            #alpha = tf.convert_to_tensor(alpha , dtype=tf.float32)
            
            #alpha is (32,32) and V_ is (32,64)        
            #print(tf.convert_to_tensor(alpha,name="alpha",dtype=tf.float32).shape)
            # for i in range(A.shape[1]):
            #     buff = 0
            #     for j in range(A.shape[3]):


            sum_PA = batch_mat_mult1(tf.transpose(V_,[0,2,1]),tf.transpose(alpha)) #check axis and directedness of graph
            #calculate h_i' before non-linearity is introduced
            sum_PA = tf.transpose(sum_PA,[0,2,1])
            # print('sum_PA',sum_PA.shape)
            h_ = tf.sigmoid(sum_PA)
            h1 = []
            if k == 0:
                h.append(h_)
                h = tf.stack(h)
                h = tf.squeeze(h,axis = 0)
            else:
                h1.append(h_)
                h1 = tf.stack(h1)
                h1 = tf.squeeze(h1,axis=0)
                h = tf.concat([h,h1],axis=2)
            print('h', h.shape)
            #print(tf.shape(V).value)
            


        return h #debug axis

'''
