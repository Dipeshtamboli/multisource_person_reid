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
from keras.layers import Dropout
from sklearn.metrics import recall_score


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


def BatchGen_p(batch_size, source1, label1, source2, label2, source3, psudo, psudo_l):
    
    total_batch = 50
    
    src1=[]
    lab1=[]
    src2=[]
    lab2=[]
    src3=[]
    lab3=[]
    
    
    la_m2=tf.argmax(label2, 1)
    with tf.Session() as sess:
        la_m2=sess.run(la_m2)
    
    for i in range(total_batch):
        uidx=[]
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
            
        src2.append(source2[uidx,:])
        lab2.append(label2[uidx,:])
        
  
        
        idx3 = rand.sample(range(1, source3.shape[0]), batch_size)
        src3.append(source3[idx3,:])
        
    
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
        
        start = start+batch_size
    total_batch_after_p = len(src1)
        
    
    
        
    
    
       
    return total_batch_after_p, src1,lab1,src2,lab2, src3
            


# In[5]:


def BatchGen(batch_size, source1, label1, source2, label2, source3):
    
    total_batch = 50
    
    src1=[]
    lab1=[]
    src2=[]
    lab2=[]
    src3=[]
    lab3=[]
    
    
    la_m2=tf.argmax(label2, 1)
    with tf.Session() as sess:
        la_m2=sess.run(la_m2)
    
    for i in range(total_batch):
        uidx=[]
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
            
        src2.append(source2[uidx,:])
        lab2.append(label2[uidx,:])
    
        idx3 = rand.sample(range(1, source3.shape[0]), batch_size)
        src3.append(source3[idx3,:])
       
    return src1,lab1,src2,lab2, src3
            


# In[6]:


# importing csv module 
import csv 


def csvToarr(filename):
    
    # initializing the titles and rows list 
    fields = [] 
    rows = [] 
  
    # reading csv file 
    with open(filename, 'r') as csvfile: 
        # creating a csv reader object 
        csvreader = csv.reader(csvfile) 
      
        # extracting field names through first row 
        fields = next(csvreader) 
  
        # extracting each data row one by one 
        for row in csvreader: 
            rows.append(row) 
  
        # get total number of rows 
    print("Total no. of rows: %d"%(csvreader.line_num))
        
    return np.asarray(rows)
    
  


# In[7]:


# csv file name 
amazon = 'amazon_amazon.csv'
digitalSlr = 'dslr_dslr.csv'
webcam = 'webcam_webcam.csv'


# In[8]:


ama_=csvToarr(amazon)
dslr_=csvToarr(digitalSlr)
web_=csvToarr(webcam)


# In[9]:


ama_l=ama_[:,-1]
ama=ama_[:,0:ama_.shape[1] -1]

dslr_l=dslr_[:,-1]
dslr=dslr_[:,0:dslr_.shape[1] -1]

web_l=web_[:,-1]
web=web_[:,0:web_.shape[1]-1]


# In[10]:


n_class = len(np.unique(ama_l))
print(n_class)


# In[ ]:





# In[11]:


ama=ama.astype(np.float32)
ama_l=ama_l.astype(np.float32)

dslr=dslr.astype(np.float32)
dslr_l=dslr_l.astype(np.float32)

web=web.astype(np.float32)
web_l=web_l.astype(np.float32)


# In[12]:


ama_ll=ama_l.reshape(ama_l.shape[0], 1)

dslr_ll=dslr_l.reshape(dslr_l.shape[0], 1)

web_ll=web_l.reshape(web_l.shape[0], 1)


# In[13]:


ama_ll


# In[14]:


amaS, amaL=aTolistC(ama, ama_ll, n_class)


# In[15]:


dslrS, dslrL=aTolistC(dslr, dslr_ll, n_class)


# In[16]:


webS, webL=aTolistC(web, web_ll, n_class)


# In[17]:


s1_l=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
s2_l=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]


# In[18]:


s11=  dslrS
s11_l=  dslrL

s22=  webS
s22_l= webL

T1= amaS
T1_l= amaL


# In[19]:


##### (source1)
sou1= splitarr(s11, 21, s1_l)
sou1_l=splitarr(s11_l, 21, s1_l)


# In[20]:


sou1_l.shape


# In[21]:


##### (source2)
sou2= splitarr(s22, 21, s2_l)
sou2_l=splitarr(s22_l, 21, s2_l)
#print(sou2_l)


# In[22]:


sou2_l = sou2_l.ravel()
sou1_l = sou1_l.ravel()


# In[23]:


sou1_l.shape


# In[24]:


hh=np.unique(sou1_l)


# In[25]:


len(hh)


# In[26]:


tar = ama
tar_ll = ama_ll


# In[27]:


tar_llk = np.copy(tar_ll)


# In[28]:


tar_llk


# In[29]:


# Target Label update
r=tar_ll.shape[0]
for i in range (r):
    
    if tar_ll[i] >= 21:
        
        tar_llk[i] = 21

        
    


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
tar_l_h=tf.one_hot(tar_l, n_classes)
with tf.Session() as sess:
    sou1_l_h=sess.run(sou1_l_h)
    sou2_l_h=sess.run(sou2_l_h)
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

X3= tf.placeholder(tf.float32, [None, n_features], name='X3' ) # Target Input data 
Y3_ind = tf.placeholder(tf.float32, [None, n_classes], name= 'Y2_ind')  # Source3 label index 
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


# In[39]:


def generator(z1, z2, z3):
    
    #with tf.variable_scope("FeatureGenerator",reuse=True):
        
    sh1_1=tf.layers.batch_normalization(tf.nn.relu(tf.matmul(z1, w1_1) + b1_1))   # source1
    #sh1_1 = Dropout(0.3)(sh1_1, training=True)
    sh2_1=tf.layers.batch_normalization(tf.nn.relu(tf.matmul(z2, w1_1) + b1_1))   # source2
    #sh2_1 = Dropout(0.3)(sh2_1, training=True)
    T1=tf.nn.relu(tf.matmul(z3, w1_1) + b1_1)    # Target
        
    sh1_2=tf.layers.batch_normalization(tf.nn.relu(tf.matmul(sh1_1, w2_1) + b2_1))   # source1
    #sh1_2 = Dropout(0.3)(sh1_2, training=True)
    sh2_2=tf.layers.batch_normalization(tf.nn.relu(tf.matmul(sh2_1, w2_1) + b2_1))   # source2
    #sh2_2 = Dropout(0.3)(sh2_2, training=True)
    T2=tf.nn.relu(tf.matmul(T1, w2_1) + b2_1)    # Target
        
    return sh1_2, sh2_2, T2

    
    
def classifier(x1, x2, t):
    
    #with tf.variable_scope("ClassClassifier", reuse = True ):
        
    X11 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(x1, w5_1) + b5_1))   # 1st layer for source1
    t11 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(t, w5_1) + b5_1))   # 1st layer target through source1
    
    p_logit_X1 = tf.matmul(X11, w6_1) + b6_1   # logit for source1
    p_logit_t1 = tf.matmul(t11, w6_1) + b6_1   # logit for target through source1
       
        
    X22 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(x2, w5_2) + b5_2))     # 2nd layer for source2
    t22 = tf.layers.batch_normalization(tf.nn.relu(tf.matmul(t, w5_2) + b5_2))   # 2nd layer target through source2
    
    p_logit_X2 = tf.matmul(X22, w6_2) + b6_2   # logit for source1
    p_logit_t2 = tf.matmul(t22, w6_2) + b6_2   # logit for target through source1
        
    p_logit_t = (p_logit_t1 + p_logit_t2)/2
        
    p_X1 = tf.nn.softmax(p_logit_X1)
        
    p_X2 = tf.nn.softmax(p_logit_X2)
        
    p_t = tf.nn.softmax(p_logit_t)
    
    p_t1 = tf.argmax(p_t,1)
        
        
    return p_logit_X1, p_logit_X2, p_logit_t, p_X1, p_X2, p_t, p_t1
        


# In[40]:


s1_f, s2_f, t_f = generator(X1, X2, X3)

logit_x1, logit_x2, logit_t, px1, px2, p_t, p_t1 = classifier(s1_f, s2_f, t_f)

t = 0.5


# In[41]:


p_loss_X1 = tf.nn.softmax_cross_entropy_with_logits(logits = logit_x1, labels = Y1_ind)
p_loss_X2 = tf.nn.softmax_cross_entropy_with_logits(logits = logit_x2, labels = Y2_ind)

p_loss_s1_s2 = tf.reduce_mean(tf.pow(s1_f - s2_f ,2), 1)



loss_cl = tf.reduce_mean(p_loss_X1) + tf.reduce_mean(p_loss_X2)

loss1 = loss_cl + p_loss_s1_s2

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

for i in range(20):
    
    hgbw = tf.cond(hing[i]<gg, lambda: hing[i], lambda:gg)
    
    bb.append(hgbw)

hinge_loss=tf.reduce_mean(tf.stack(bb)) # hingh loss

reg1 = tf.norm(w5_1,2) + tf.norm(w5_2,2) + tf.norm(w6_1,2) + tf.norm(w6_2,2)
reg2 = tf.norm(w1_1,2) + tf.norm(w2_1,2)


loss2 = loss_cl + Ladv  # classifier

loss3 = loss1 - Ladv - hinge_loss  # Generator


# In[ ]:





# In[42]:


G_v = [w1_1, w2_1, b1_1, b2_1]
C_v = [w5_1, w5_2, b5_1, b5_2, w6_1, w6_2, b6_1, b6_2]


# In[43]:


#loss_cl_op = tf.train.AdamOptimizer(0.001).minimize(loss1)

loss1C_op = tf.train.AdamOptimizer(0.001).minimize(loss2, var_list = C_v)

loss2G_op = tf.train.AdamOptimizer(0.001).minimize(loss3, var_list = G_v)


# In[44]:


p1_acc =tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y1_ind, 1), tf.argmax(px1, 1)), tf.float32))

p2_acc =tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y2_ind, 1), tf.argmax(px2, 1)), tf.float32))

pt_acc =tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y3_ind, 1), tf.argmax(p_t, 1)), tf.float32))


# In[45]:


init=tf.global_variables_initializer()


# In[47]:


num_epoch=100
batch_size=32
dp1 = 0.0
dp2 = 0.0
dp11 = 0.0
dp12 = 0.0
dp13 = 0.0

l1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
l2 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
#ps_t = []
#ps_t_l = []

flag = 0



with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
        
        ps_t = []
        ps_t_l = []
        
        if flag ==0:
            total_batches = 50
            
            X1_batch,y1_batch, X2_batch,y2_batch, X3_batch=BatchGen(batch_size, sou1, sou1_l_h, sou2, sou2_l_h, tar)
        else:
            total_batches, X1_batch,y1_batch, X2_batch,y2_batch, X3_batch=BatchGen_p(batch_size, sou1, sou1_l_h, sou2,sou2_l_h, tar, psudo, psudo_l)
        
        flag = 0

        for batch in range(total_batches):
            
            xx_1=X1_batch[batch]
            #print(xx_1.shape)
            
            xx_2=X2_batch[batch]
            #print(xx_2.shape)
            
            xx_3=X3_batch[batch]
            #print(xx_3.shape)
            
            yy_1=y1_batch[batch]
            yy_2=y2_batch[batch]
            

            
            #_,batch_loss_cl  = sess.run([loss_cl_op, loss1],
                                        #feed_dict={X1: xx_1, X2:xx_2, Y1_ind:yy_1, Y2_ind:yy_2, X3:xx_3})
            
            _,batch_loss1C, pu_one_hot, pu_logit, pu_argmax  = sess.run([loss1C_op, loss2, pu_l_h, jh, kh],
                                        feed_dict={X1: xx_1, X2:xx_2, Y1_ind:yy_1, Y2_ind:yy_2, X3:xx_3})
            
                    
            
            
            _,batch_loss2G  = sess.run([loss2G_op, loss3],feed_dict={X1: xx_1, X2:xx_2, Y1_ind:yy_1, Y2_ind:yy_2, X3:xx_3})
            
            rr=0
            for v in range(pu_one_hot.shape[0]):
                
                ff = pu_one_hot[v]
                aa=pu_argmax[v]
                dwq = tf.cast(ff, tf.int32)
               
                if pu_logit[rr, aa] >= 0.95:
                    flag = 1
                        
                    ps_t.append(xx_3[v, :])
                    ps_t_l.append(ff)
                rr=rr+1

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
            pass_avg = sess.run([pt_acc], feed_dict={X3:tar, Y3_ind:tar_l_h})
            lab = sess.run([p_t1], feed_dict={X3:tar, Y3_ind:tar_l_h})
            
            
            lab = np.matrix.transpose(np.asarray(lab))

            OS = recall_score(tar_l, lab, labels=l2, average='macro')
            OS_star = recall_score(tar_l, lab, labels=l1, average='macro')

       
            print('target from source1', pas11)
            print('target from source2', pas12)
            print('OS', OS)
            print('OS-star', OS_star)
            print('target average', pass_avg)
        
        if OS >= dp11:
            
            dp11=OS
            dp12=OS_star
            dp13 = pass_avg[0]
            dp1 = pas11[0]
            dp2 = pas12[0]
            ep11=epoch

      
        print('print highest source1: %f, epoch%f'%(dp1, ep11))
        print('print highest source2: %f, epoch%f'%(dp2, ep11))
        print('print highest OS: %f, epoch%f'%(dp11, ep11))
        print('print highest OS-star: %f, epoch%f'%(dp12, ep11))
        print('print highest ALL: %f, epoch%f'%(dp13, ep11))
                            
    print("Training Complete")
    
    sess.close()            


# In[ ]:





# In[ ]:




