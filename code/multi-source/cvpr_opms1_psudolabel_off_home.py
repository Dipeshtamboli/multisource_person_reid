#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import scipy.io as sio
import sklearn as sk
import time
import random as rand
from random import randrange
#from numpy import array
#from numpy.linalg import norm
from flip_gradient import flip_gradient
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csgraph
import os
import matplotlib.pyplot as plt
from random import *
from sklearn.metrics import recall_score
from keras.layers import Dropout


# In[2]:


def splitarr(slist, k, index):
    p=index[0]
    st1=np.asarray(slist[p])
    for i in range(k-1):
        
        q=index[i+1]
        
        st2=np.asarray(slist[q])
        
        ar=np.vstack((st1, st2))
        
        st1=ar
        
    return ar


# In[3]:


def aTolistC(array, class_l, n_classes):
    
    slist=[]
    l_list=[]
    
    for j in range(n_classes):
        k=[i for i,x in enumerate(class_l) if x == j]
        
        slist.append(array[k,:])
        l_list.append(class_l[k])
        
    return slist, l_list
    
    


# In[4]:


def BatchGen_p(batch_size, source1, label1, source2, label2, source3, label3,source4, psudo, psudo_l):
    
    total_batch = 25
    
    src1=[]
    lab1=[]
    src2=[]
    lab2=[]
    src3=[]
    lab3=[]
    src4=[]
    
    
    la_m2=tf.argmax(label2, 1)
    la_m3=tf.argmax(label3, 1)
    with tf.Session() as sess:
        la_m2=sess.run(la_m2)
        la_m3=sess.run(la_m3)
    
    for i in range(total_batch):
        uidx=[]
        uidx1=[]
        idx1 = rand.sample(range(1, source1.shape[0]), batch_size)
        src1.append(source1[idx1,:])
        lab1.append(label1[idx1,:])
        lab=label1[idx1,:]
        la_m=tf.argmax(label1[idx1,:], 1)
        with tf.Session() as sess:
            la_m=sess.run(la_m)
        for j in range(batch_size):
            p=la_m[j]
            #print(p)
            result = [index for index, word in enumerate(la_m2) if word == p]
            q=randint(0, len(result)-1)
            #print(result[q])
            uidx.append(result[q])
            
        for jj in range(batch_size):
            p=la_m[jj]
            result1 = [index for index, word in enumerate(la_m3) if word == p]
            q1=randint(0, len(result1)-1)
            uidx1.append(result1[q1])
            
            
            
        src2.append(source2[uidx,:])
        lab2.append(label2[uidx,:])
        
        src3.append(source3[uidx1,:])
        lab3.append(label3[uidx1,:])
        
  
        
        idx3 = rand.sample(range(1, source4.shape[0]), batch_size)
        src4.append(source4[idx3,:])
        
    
    psudo_renge = int(psudo.shape[0]/batch_size)
    
    #print("kkkkkkkkkkkkkkkkkkkkkkkkkkk")
    #print(psudo)
    #print("hhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    #print(psudo[0])
    #print("lllllllllllllllllllllllll")
    
    #print(psudo[0:5])
    
    start = 0
    for i in range(psudo_renge):
        psu = psudo[start:start+batch_size]
        psu_l = psudo_l[start:start+batch_size]
        
        src1.append(psu)
        lab1.append(psu_l)
        
        src2.append(psu)
        
        lab2.append(psu_l)
        
        src3.append(psu)
        
        lab3.append(psu_l)
        
        src4.append(psu)
        
        start = start+batch_size
    total_batch_after_p = len(src1)
        
    
    
        
    
    
       
    return total_batch_after_p, src1,lab1,src2,lab2, src3, lab3, src4
            


# In[5]:


def BatchGen(batch_size, source1, label1, source2, label2, source3, label3, source4):
    
    total_batch = 25
    
    src1=[]
    lab1=[]
    src2=[]
    lab2=[]
    src3=[]
    lab3=[]
    lab3=[]
    src4=[]
    
    
    la_m2=tf.argmax(label2, 1)
    la_m3=tf.argmax(label3, 1)
    
    with tf.Session() as sess:
        la_m2=sess.run(la_m2)
        la_m3=sess.run(la_m3)
    
    for i in range(total_batch):
        uidx=[]
        uidx1=[]
        idx1 = rand.sample(range(1, source1.shape[0]), batch_size)
        src1.append(source1[idx1,:])
        lab1.append(label1[idx1,:])
        lab=label1[idx1,:]
        la_m=tf.argmax(label1[idx1,:], 1)
        with tf.Session() as sess:
            la_m=sess.run(la_m)
        for j in range(batch_size):
            p=la_m[j]
            #print(p)
            result = [index for index, word in enumerate(la_m2) if word == p]
            q=randint(0, len(result)-1)
            #print(result[q])
            uidx.append(result[q])
            
        for jj in range(batch_size):
            p=la_m[jj]
            result1 = [index for index, word in enumerate(la_m3) if word == p]
            q1=randint(0, len(result1)-1)
            uidx1.append(result1[q1])
            
            
        src2.append(source2[uidx,:])
        lab2.append(label2[uidx,:])
        
        src3.append(source3[uidx1,:])
        lab3.append(label3[uidx1,:])
        
        
    
        idx4 = rand.sample(range(1, source4.shape[0]), batch_size)
        src4.append(source4[idx4,:])
       
    return src1,lab1,src2,lab2, src3, lab3, src4
            


# In[8]:


# csv file name 
art_ = sio.loadmat('OffHoRes/art/a_f.mat')
art_l_ = sio.loadmat('OffHoRes/art/a_l.mat')

clipart_ = sio.loadmat('OffHoRes/clipart/c_f.mat')
clipart_l_ = sio.loadmat('OffHoRes/clipart/c_l.mat')

product_ = sio.loadmat('OffHoRes/product/p_f.mat')
product_l_ = sio.loadmat('OffHoRes/product/p_l.mat')

real_world_ = sio.loadmat('OffHoRes/real_world/r_f.mat')
real_world_l_ = sio.loadmat('OffHoRes/real_world/r_l.mat')


# In[9]:


art = art_['a_f']
art_l = art_l_['a_l']
#art_l=art_l.ravel()

clipart = clipart_['c_f']
clipart_l = clipart_l_['c_l']
#clipart_l = clipart_l.ravel()

product = product_['p_f']
product_l = product_l_['p_l']
#product_l = product_l.ravel()

real_world = real_world_['r_f']
real_world_l = real_world_l_['r_l']
#real_world_l = real_world_l.ravel()


# In[10]:


art_l


# In[11]:


n_class = len(np.unique(art_l))
print(n_class)


# In[ ]:





# In[12]:


art=art.astype(np.float32)
art_l=art_l.astype(np.float32)

clipart=clipart.astype(np.float32)
clipart_l=clipart_l.astype(np.float32)

product=product.astype(np.float32)
product_l=product_l.astype(np.float32)

real_world=real_world.astype(np.float32)
real_world_l=real_world_l.astype(np.float32)


# In[13]:


artS, artL=aTolistC(art, art_l, n_class)


# In[14]:


clipartS, clipartL=aTolistC(clipart, clipart_l, n_class)


# In[15]:


productS, productL=aTolistC(product, product_l, n_class)


# In[16]:


real_worldS, real_worldL=aTolistC(real_world, real_world_l, n_class)


# In[17]:


s1_l=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
      29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]
s2_l=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
      29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]

s3_l=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
      29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]


# In[18]:


s11=  artS
s11_l=  artL

s22=  productS
s22_l= productL

s33=  clipartS
s33_l= clipartL

tar = real_world
tar_ll = real_world_l


# In[19]:


##### (source1)
sou1= splitarr(s11, 45, s1_l)
sou1_l=splitarr(s11_l, 45, s1_l)


# In[20]:


sou1_l.shape


# In[22]:


##### (source2)
sou2= splitarr(s22, 45, s2_l)
sou2_l=splitarr(s22_l, 45, s2_l)
#print(sou2_l)


# In[23]:


##### (source3)
sou3= splitarr(s33, 45, s2_l)
sou3_l=splitarr(s33_l, 45, s2_l)
#print(sou2_l)


# In[24]:


sou2_l = sou2_l.ravel()
sou1_l = sou1_l.ravel()
sou3_l = sou3_l.ravel()


# In[25]:


hh=np.unique(sou1_l)


# In[26]:


len(hh)


# In[27]:


tar_llk = np.copy(tar_ll)


# In[28]:


tar_llk


# In[29]:


# Target Label update
r=tar_ll.shape[0]
for i in range (r):
    
    if tar_ll[i] >= 45:
        
        tar_llk[i] = 45

        
    


# In[30]:


n_classes = len(np.unique(tar_llk))
print(n_classes)
tar_llk = tar_llk.ravel()
#web_llk


# In[31]:


tar_l = tar_llk


# In[32]:


sou1_l_h=tf.one_hot(sou1_l, n_classes)
sou2_l_h=tf.one_hot(sou2_l, n_classes)
sou3_l_h=tf.one_hot(sou3_l, n_classes)
tar_l_h=tf.one_hot(tar_l, n_classes)
with tf.Session() as sess:
    sou1_l_h=sess.run(sou1_l_h)
    sou2_l_h=sess.run(sou2_l_h)
    sou3_l_h=sess.run(sou3_l_h)
    tar_l_h=sess.run(tar_l_h)
#print(tar_l_h[0])


# In[33]:


#a,b,c,d,e,g = BatchGen(10, sou1,sou1_l_h, sou2, sou2_l_h, web, tar_l_h )


# In[ ]:





# In[34]:


def weight_variable(shape):
    
    
    initial = tf.random_normal(shape, stddev=0.1)
    #initial = tf.truncated_normal(shape)#, stddev=0.1)
    return tf.Variable(initial)

def bise_variable(shape):
   
    initial = tf.random_normal(shape)
    #initial = tf.constant(0.1, "float32", shape)
    return tf.Variable(initial)


# In[35]:


n_features=sou1.shape[1]

X1= tf.placeholder(tf.float32, [None, n_features], name='X1' ) # Source1 Input data 
Y1_ind = tf.placeholder(tf.float32, [None, n_classes], name= 'Y1_ind')  # Source1 label index 

X2= tf.placeholder(tf.float32, [None, n_features], name='X2' ) # Source2 Input data 
Y2_ind = tf.placeholder(tf.float32, [None, n_classes], name= 'Y2_ind')  # Source2 label index 

X3= tf.placeholder(tf.float32, [None, n_features], name='X3' ) # source3 Input data 
Y3_ind = tf.placeholder(tf.float32, [None, n_classes], name= 'Y2_ind')  # Source3 label index 

X4= tf.placeholder(tf.float32, [None, n_features], name='X4' ) # Target Input data 
Y4_ind = tf.placeholder(tf.float32, [None, n_classes], name= 'Y4_ind')  # target label index 


l = tf.placeholder(tf.float32, [], name= 'l')  # gradient reversal layer


# In[36]:


num_inputs=n_features    
num_hid1=1200
num_hid2=600
num_hid3=128


# In[37]:


w1_1=tf.Variable(tf.random_normal([num_inputs, num_hid1], stddev = 0.01), name= 'w1_1')
b1_1=tf.Variable(tf.random_normal([num_hid1]), name = 'b1_1')

w1_2=tf.Variable(tf.random_normal([num_inputs, num_hid1], stddev = 0.01), name= 'w1_2')
b1_2=tf.Variable(tf.random_normal([num_hid1]), name = 'b1_2')

w2_1=tf.Variable(tf.random_normal([num_hid1, num_hid2], stddev = 0.01), name= 'w2_1')
b2_1=tf.Variable(tf.random_normal([num_hid2]), name = 'b2_1')

w2_2=tf.Variable(tf.random_normal([num_hid1, num_hid2]), name= 'w2_2')
b2_2=tf.Variable(tf.random_normal([num_hid2]), name = 'b2_2')


# In[38]:


w5_1=tf.Variable(tf.random_normal([num_hid2, num_hid3], stddev = 0.01), name= 'w5_1')
b5_1=tf.Variable(tf.random_normal([num_hid3]), name = 'b5_1')

w6_1=tf.Variable(tf.random_normal([num_hid3, n_classes], stddev = 0.01), name= 'w6_1')
b6_1=tf.Variable(tf.random_normal([n_classes]), name = 'b6_1')



w5_2=tf.Variable(tf.random_normal([num_hid2, num_hid3], stddev = 0.01), name= 'w5_2')
b5_2=tf.Variable(tf.random_normal([num_hid3]), name = 'b5_2')

w6_2=tf.Variable(tf.random_normal([num_hid3, n_classes], stddev = 0.01), name= 'w6_2')
b6_2=tf.Variable(tf.random_normal([n_classes]), name = 'b6_2')

w5_3=tf.Variable(tf.random_normal([num_hid2, num_hid3], stddev = 0.01), name= 'w5_3')
b5_3=tf.Variable(tf.random_normal([num_hid3]), name = 'b5_3')

w6_3=tf.Variable(tf.random_normal([num_hid3, n_classes], stddev = 0.01), name= 'w6_3')
b6_3=tf.Variable(tf.random_normal([n_classes]), name = 'b6_3')



# In[39]:


def generator(z1, z2, z3, z4):
    
    #with tf.variable_scope("FeatureGenerator",reuse=True):
        
    sh1_1=tf.layers.batch_normalization(tf.nn.relu(tf.matmul(z1, w1_1) + b1_1))   # source1
    #sh1_1 = Dropout(0.3)(sh1_1, training=True)
    sh2_1=tf.layers.batch_normalization(tf.nn.relu(tf.matmul(z2, w1_1) + b1_1))   # source2
    #sh2_1 = Dropout(0.3)(sh2_1, training=True)
    
    sh3_1=tf.layers.batch_normalization(tf.nn.relu(tf.matmul(z3, w1_1) + b1_1))   # source3
    
    T1=tf.nn.relu(tf.matmul(z4, w1_1) + b1_1)    # Target
        
    sh1_2=tf.layers.batch_normalization(tf.nn.relu(tf.matmul(sh1_1, w2_1) + b2_1))   # source1
    #sh1_2 = Dropout(0.3)(sh1_2, training=True)
    sh2_2=tf.layers.batch_normalization(tf.nn.relu(tf.matmul(sh2_1, w2_1) + b2_1))   # source2
    #sh2_2 = Dropout(0.3)(sh2_2, training=True)
    
    sh3_2=tf.layers.batch_normalization(tf.nn.relu(tf.matmul(sh3_1, w2_1) + b2_1))   # source3
    #sh2_2 = Dropout(0.3)(sh2_2, training=True)
    
    T2=tf.nn.relu(tf.matmul(T1, w2_1) + b2_1)    # Target
        
    return sh1_2, sh2_2, sh3_2, T2

    
    
def classifier(x1, x2, x3, t):
    
    #with tf.variable_scope("ClassClassifier", reuse = True ):
        
    X11 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(x1, w5_1) + b5_1))   # 1st layer for source1
    t11 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(t, w5_1) + b5_1))   # 1st layer target through source1
    
    p_logit_X1 = tf.matmul(X11, w6_1) + b6_1   # logit for source1
    p_logit_t1 = tf.matmul(t11, w6_1) + b6_1   # logit for target through source1
       
        
    X22 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(x2, w5_2) + b5_2))     # 2nd layer for source2
    t22 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(t, w5_2) + b5_2))   # 2nd layer target through source2
    
    p_logit_X2 = tf.matmul(X22, w6_2) + b6_2   # logit for source2
    p_logit_t2 = tf.matmul(t22, w6_2) + b6_2   # logit for target through source2
        
        
    X33 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(x3, w5_3) + b5_3))     # 2nd layer for source3
    t33 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(t, w5_3) + b5_3))   # 2nd layer target through source3
    
    p_logit_X3 = tf.matmul(X33, w6_3) + b6_3   # logit for source3
    p_logit_t3 = tf.matmul(t33, w6_3) + b6_3   # logit for target through source3
        
    
    p_logit_t = (p_logit_t1 + p_logit_t2 + p_logit_t3)/3
        
    p_X1 = tf.nn.softmax(p_logit_X1)
        
    p_X2 = tf.nn.softmax(p_logit_X2)
    
    p_X3 = tf.nn.softmax(p_logit_X3)
        
    p_t = tf.nn.softmax(p_logit_t)
    
    p_t1 = tf.argmax(p_t)
        
        
    return p_logit_X1, p_logit_X2, p_logit_X3, p_logit_t, p_X1, p_X2, p_X3, p_t, p_t1
        


# In[40]:


s1_f, s2_f, s3_f, t_f = generator(X1, X2, X3, X4)

logit_x1, logit_x2, logit_x3, logit_t, px1, px2, px3, p_t, p_t1 = classifier(s1_f, s2_f, s3_f, t_f)

t = 0.5


# In[62]:


p_loss_X1 = tf.nn.softmax_cross_entropy_with_logits(logits = logit_x1, labels = Y1_ind)
p_loss_X2 = tf.nn.softmax_cross_entropy_with_logits(logits = logit_x2, labels = Y2_ind)
p_loss_X3 = tf.nn.softmax_cross_entropy_with_logits(logits = logit_x3, labels = Y3_ind)

p_loss_s1_s2 = tf.reduce_mean(tf.pow(s1_f - s2_f ,2), 1)
p_loss_s3_s2 = tf.reduce_mean(tf.pow(s3_f - s2_f ,2), 1)
p_loss_s1_s3 = tf.reduce_mean(tf.pow(s1_f - s3_f ,2), 1)





loss_cl = tf.reduce_mean(p_loss_X1) + tf.reduce_mean(p_loss_X2) +tf.reduce_mean(p_loss_X3)

loss1 = loss_cl + p_loss_s1_s2 + p_loss_s3_s2 +p_loss_s1_s3

p_kone=tf.gather(p_t,indices=[n_classes-1],axis=1)
Ladv=-0.5*tf.reduce_mean(tf.log(p_kone+1e-8))-0.5*tf.reduce_mean((tf.log(1.0-(p_kone)+1e-8)))

jh= p_t[:, :n_classes-1]

kh=tf.argmax(jh, axis = 1)
pu_l_h=tf.one_hot(kh, n_classes)

gbp=p_t[:, -1]
hg = tf.reduce_mean(jh, 1)
alpha = 0.2
hing=tf.abs(p_t[:, -1] - hg) - alpha

row = jh.shape[0]
gg=0.0

bb=[]

for i in range(2):
    
    hgbw = tf.cond(hing[i]<gg, lambda: hing[i], lambda:gg)
    
    bb.append(hgbw)

hingh_loss=tf.reduce_mean(tf.stack(bb))  # hingh loss

loss2 = loss_cl + Ladv   # classifier

loss3 = loss1 - Ladv - hingh_loss   # Generator


# In[ ]:





# In[63]:


G_v = [w1_1, w2_1, b1_1, b2_1]
C_v = [w5_1, w5_2, w5_3, b5_1, b5_2, b5_3, w6_1, w6_2, w6_3, b6_1, b6_2,b6_3]


# In[64]:


#loss_cl_op = tf.train.AdamOptimizer(0.001).minimize(loss1)

loss1C_op = tf.train.AdamOptimizer(0.001).minimize(loss2, var_list = C_v)

loss2G_op = tf.train.AdamOptimizer(0.001).minimize(loss3, var_list = G_v)


# In[65]:


p1_acc =tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y1_ind, 1), tf.argmax(px1, 1)), tf.float32))

p2_acc =tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y2_ind, 1), tf.argmax(px2, 1)), tf.float32))

p3_acc =tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y3_ind, 1), tf.argmax(px3, 1)), tf.float32))

pt_acc =tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y4_ind, 1), tf.argmax(p_t, 1)), tf.float32))


# In[66]:


init=tf.global_variables_initializer()


# In[72]:


num_epoch=100
batch_size=20
dp1 = 0.0
dp2 = 0.0
dp3=0.0
dp11 = 0.0
dp12 = 0.0
dp13=0.0

#ps_t = []
#ps_t_l = []

flag = 0

l1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]
l2=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
        
        ps_t = []
        ps_t_l = []
        
        if flag ==0:
            total_batches = 25
            
            X1_batch,y1_batch, X2_batch,y2_batch, X3_batch, y3_batch, X4_batch=BatchGen(batch_size, sou1, sou1_l_h,
                                                                    sou2, sou2_l_h, sou3, sou3_l_h, tar)
        else:
            total_batches, X1_batch,y1_batch, X2_batch,y2_batch, X3_batch, y3_batch, X4_batch=BatchGen_p(batch_size, sou1,
                                                                                     sou1_l_h, sou2,sou2_l_h,
                                                                                     sou3, sou3_l_h,
                                                                                     tar, psudo, psudo_l)
        
        flag = 0

        for batch in range(total_batches):
            
            xx_1=X1_batch[batch]
            #print(xx_1.shape)
            
            xx_2=X2_batch[batch]
            #print(xx_2.shape)
            
            xx_3=X3_batch[batch]
            #print(xx_3.shape)
            
            xx_4=X4_batch[batch]
            
            yy_1=y1_batch[batch]
            yy_2=y2_batch[batch]
            yy_3=y3_batch[batch]
                        

            
            #_,batch_loss_cl  = sess.run([loss_cl_op, loss1],
                                        #feed_dict={X1: xx_1, X2:xx_2, Y1_ind:yy_1, Y2_ind:yy_2, X3:xx_3})
            
            _,batch_loss1C, pu_one_hot, pu_logit, pu_argmax  = sess.run([loss1C_op, loss2, pu_l_h, jh, kh],
                                        feed_dict={X1: xx_1, X2:xx_2, Y1_ind:yy_1, Y2_ind:yy_2,
                                                   X3:xx_3, Y3_ind:yy_3, X4:xx_4})
            
            
            rr=0
            for v in range(pu_one_hot.shape[0]):
                
                ff = pu_one_hot[v]
                aa=pu_argmax[v]
                dwq = tf.cast(ff, tf.int32)
               
                if pu_logit[rr, aa] >= 0.9:
                    flag = 1
                        
                    ps_t.append(xx_4[v, :])
                    ps_t_l.append(ff)
                    
    
                rr=rr+1
                    
            
            
            _,batch_loss2G  = sess.run([loss2G_op, loss3],
                                        feed_dict={X1: xx_1, X2:xx_2, Y1_ind:yy_1, Y2_ind:yy_2, X3:xx_3,
                                                   Y3_ind:yy_3, X4:xx_4})
            
        #ps_t.append(xx_1)
        #ps_t_l.append(yy_1)
        psudo=np.asarray(ps_t)
        psudo_l = np.asarray(ps_t_l)
        
        
        print(len(ps_t))
        p = epoch%1
        
        if p==0:
            print("epoch#####555555555555##########################", epoch)
                    
            pas11 = sess.run([p1_acc], feed_dict={X1:tar, Y1_ind:tar_l_h})
            pas12 = sess.run([p2_acc], feed_dict={X2:tar, Y2_ind:tar_l_h})
            pas13 = sess.run([p3_acc], feed_dict={X3:tar, Y3_ind:tar_l_h})
            lab = sess.run([p_t1], feed_dict={X4:tar, Y4_ind:tar_l_h})
            pass_avg2 = sess.run([pt_acc], feed_dict={X4:tar, Y4_ind:tar_l_h})
            
            lab = np.matrix.transpose(np.asarray(lab))
            #print(tar_l.shape)
            print(lab.shape)

            OS = recall_score(tar_l, lab, labels=l2, average='macro')
            OS_star = recall_score(tar_l, lab, labels=l1, average='macro')

            print('target from source1', pas11)
            print('target from source2', pas12)
            print('target from source3', pas13)
            print('OS', OS)
            print('OS-star', OS_star)
            print('OS-all', pass_avg2)
        
        if OS >= dp11:
            
            dp11 = OS
            dp12 = OS_star
            dp13 = pass_avg2
            
            dp1 = pas11
            dp2 = pass12
            dp3 = pass13
            
      
        print('OS: %f OS-star: %f Avg: %f' %(dp11, dp12, dp13))

                            
                            
    print("Training Complete")
    
    sess.close()            

