# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base
from torch.autograd import Variable
from path import MODEL_PATH
from vgg import VGG
from utils import labels,COCO_INSTANCE_CATEGORY_NAMES
import torchvision
from PIL import Image
from torchvision import transforms as T
from flyai.processor.download import check_download


from path import DATA_PATH

#__import__('net', fromlist=["Net"])

Torch_MODEL_NAME = "model.pkl"

cuda_avail = torch.cuda.is_available()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
def get_prediction(pred, threshold):
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class
def decode_pred_bb(pred):
    results = []
    for bb,lab in zip(pred[0],pred[1]):
        try:
            results.append([labels.index(lab),bb[0][0],bb[0][1],bb[1][0],bb[1][1]])
        except:
            print("no label",lab)
    return results


    
class Model(Base):
    def __init__(self, data):
        self.data = data

    def predict(self, **data):
        cnn = torch.load(os.path.join(MODEL_PATH, Torch_MODEL_NAME))
        if cuda_avail:
            cnn.cuda()
        x_data = self.data.predict_data(**data)
        x_data = torch.from_numpy(x_data)
        x_data = x_data.float()
        if cuda_avail:
            x_data = Variable(x_data.cuda())
        
        outputs = cnn(x_data)
        outputs = outputs.cpu()
        prediction = get_prediction(outputs,0.5)
        prediction = decode_pred_bb(prediction)
        return prediction

    def predict_all(self, datas):
        print(os.path.join(MODEL_PATH, Torch_MODEL_NAME))
        #cnn = torch.load(os.path.join(MODEL_PATH, Torch_MODEL_NAME))
        #cnn = Net()
        
        cnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        cnn.load_state_dict(torch.load(os.path.join(MODEL_PATH, Torch_MODEL_NAME)))
        cnn.eval()
        cnn.to(device)
        
        if cuda_avail:
            cnn.cuda()
        labels = []
        for data in datas:
            img = Image.open(check_download(data['image_path'], DATA_PATH)) # Load the image
            img = img.resize((800,800))
            transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
            x_data = transform(img).cuda() # Apply the transform to the image
            outputs = cnn([x_data])
            #print(outputs)
            try:
                prediction = get_prediction(outputs,0.5)
                #print(prediction)
                prediction = decode_pred_bb(prediction)
                labels.append(prediction)
            except:
                prediction = get_prediction(outputs,0.1)
                #print(prediction)
                prediction = decode_pred_bb(prediction)
                labels.append(prediction)
        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=Torch_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        #torch.save(network, os.path.join(path, name))
        torch.save(network.state_dict(), os.path.join(path, name))


