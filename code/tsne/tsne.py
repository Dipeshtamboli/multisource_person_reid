'''
			Duke	MARKET 		
train_all	702		751			
train		702		751			
gallery		1110	752			
query		702		750			
val			702		751			
multi-query	---		1501		
'''
import pdb
import numpy as np
from scipy import io
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import time
start_time = time.time()

market = "/home/dipesh/multisource_person_reid/code/pytorch_result.mat"
duke = "/home/dipesh/multisource_person_reid/code/pytorch_result_duke.mat"

d_m = "/home/dipesh/multisource_person_reid/code/pytorch_result_d-m.mat"
m_d = "/home/dipesh/multisource_person_reid/code/pytorch_result_m-d.mat"

market = io.loadmat(market)
duke = io.loadmat(duke)
d_m = io.loadmat(d_m)
m_d = io.loadmat(m_d)

market_f = market["gallery_f"]
duke_f = duke["gallery_f"]
d_m_f = d_m["gallery_f"]
m_d_f = m_d["gallery_f"]

tsne = TSNE(n_jobs=16)

num_market = market_f.shape[0]
print("num_market",num_market)
num_duke = duke_f.shape[0]
print("num_duke",num_duke)
num_d_m = d_m_f.shape[0]
print("num_d_m",num_d_m)
num_m_d = m_d_f.shape[0]
print("num_m_d",num_m_d)

all_features = np.concatenate((market_f,duke_f,d_m_f,m_d_f), axis = 0)

dataset_label = np.zeros((all_features.shape[0],1))

dataset_label[num_market:num_market + num_duke] = 1
dataset_label[num_market + num_duke:num_market + num_duke + num_d_m] = 2
dataset_label[num_market + num_duke + num_d_m:] = 3

# all_features = all_features[:5]
# dataset_label = dataset_label[:5]
print("market_f.shape",market_f.shape)
print("duke_f.shape",duke_f.shape)
print("d_m_f.shape",d_m_f.shape)
print("m_d_f.shape",m_d_f.shape)
print("all_features.shape",all_features.shape)
print("(dataset_label[:,0].shape)",(dataset_label[:,0].shape))
# print("sum(dataset_label)",sum(dataset_label))
# exit()

embeddings = tsne.fit_transform(all_features)

# pdb.set_trace()

vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 4)

plot = sns.scatterplot(vis_x, vis_y, hue=dataset_label[:,0], legend='full', palette=palette)
plt.savefig("all_sns_tsne.png")
print("--- {} mins {} secs---".format((time.time() - start_time)//60,(time.time() - start_time)%60))
