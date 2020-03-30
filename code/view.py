from scipy import io

print(" ########### market 1501 ###################")
pytorch_result = io.loadmat('pytorch_result_duke.mat')

gallery_f = pytorch_result["gallery_f"]
gallery_label = pytorch_result["gallery_label"]
gallery_cam = pytorch_result["gallery_cam"]
query_f = pytorch_result["query_f"]
query_label = pytorch_result["query_label"]
query_cam = pytorch_result["query_cam"]
train_all_f = pytorch_result["train_all_f"]
train_all_label = pytorch_result["train_all_label"]
train_all_cam = pytorch_result["train_all_cam"]

print("gallery_f",gallery_f.shape)
print("gallery_label",gallery_label.shape)
print("gallery_cam",gallery_cam.shape)
print("query_f",query_f.shape)
print("query_label",query_label.shape)
print("query_cam",query_cam.shape)
print("train_all_f",train_all_f.shape)
print("train_all_label",train_all_label.shape)
print("train_all_cam",train_all_cam.shape)

print(" ########### market 1501 ###################")
pytorch_result = io.loadmat('pytorch_result.mat')


gallery_f = pytorch_result["gallery_f"]
gallery_label = pytorch_result["gallery_label"]
gallery_cam = pytorch_result["gallery_cam"]
query_f = pytorch_result["query_f"]
query_label = pytorch_result["query_label"]
query_cam = pytorch_result["query_cam"]
train_all_f = pytorch_result["train_all_f"]
train_all_label = pytorch_result["train_all_label"]
train_all_cam = pytorch_result["train_all_cam"]

print("gallery_f",gallery_f.shape)
print("gallery_label",gallery_label.shape)
print("gallery_cam",gallery_cam.shape)
print("query_f",query_f.shape)
print("query_label",query_label.shape)
print("query_cam",query_cam.shape)
print("train_all_f",train_all_f.shape)
print("train_all_label",train_all_label.shape)
print("train_all_cam",train_all_cam.shape)
