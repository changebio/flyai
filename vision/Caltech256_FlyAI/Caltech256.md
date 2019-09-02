# 图像分类: Caltech 256

Caltech 256数据集是加利福尼亚理工学院收集整理的数据集，该数据集选自Google Image 数据集，并手工去除了不符合其类别的图片。在该数据集中，图片被分为256类，每个类别的图片超过80张。

本项目使用在PyTorch框架下搭建的神经网络来完成图片分类的任务。由于网络输出的类别数量很大，简单的网络模型无法达到很好的分类效果，因此，本项目使用了预训练的densenet121模型，并仅训练全连接层的参数。

## 项目流程

1. 数据处理
2. Densenet模型解读
3. 加载预训练网络模型
4. 训练神经网络

## 数据处理

首先从指定路径读取图像，将图像大小更改为224*224，并将图片范围从0-255改为0-1：

```python
from PIL import Image
image = Image.open(path)
image=image.resize((224,224))
x_data = x_data.astype(numpy.float32)
x_data = numpy.multiply(x_data, 1.0 / 255.0)  ## scale to [0,1] from [0,255]
```

由于此数据集中有少量图片的色彩是单通道的，而神经网络的输入需要为三个通道，因此，将该通道的数据复制到三个通道上：

```python
if len(x_data.shape)!=3:
temp = numpy.zeros((x_data.shape[0],x_data.shape[1],3))
temp[:,:,0] = x_data
temp[:,:,1] = x_data
temp[:,:,2] = x_data
x_data = temp
x_data=numpy.transpose(x_data,(2,0,1)) ## reshape 
```

在上述步骤之后，对图片进行白化，即让像素点的平均值为0，方差为1。这样做是为了减小图片的范围，使得图片的特征更易于学习。白化的过程如下所示：

```python
if x_train is not None:
    x_train[:, 0, :, :] = (x_train[:, 0, :, :] - 0.485) / 0.229
    x_train[:, 1, :, :] = (x_train[:, 1, :, :] - 0.456) / 0.224
    x_train[:, 2, :, :] = (x_train[:, 2, :, :] - 0.406) / 0.225

if x_test is not None:
    x_test[:, 0, :, :] = (x_test[:, 0, :, :] - 0.485) / 0.229
    x_test[:, 1, :, :] = (x_test[:, 1, :, :] - 0.456) / 0.224
    x_test[:, 2, :, :] = (x_test[:, 2, :, :] - 0.406) / 0.225
```

## DenseNet模型解读

DenseNet的网络结构如下图所示。在传统的CNN中，每个卷积层只与其相邻的卷积层相连接，这就造成了位于网络浅层的参数在反向传播中获取的梯度非常小，也就是梯度消失问题。

DenseNet设计了名为Dense Block的特殊的网络结构，在一个Dense Block中，每个层的输入为前面所有层的输出，这也正是Dense的含义。通过这种方法，在反向传播中，网络浅层的参数可以从后面所有层中获得梯度，在很大程度上减弱了梯度消失的问题。值得注意的是，每个层只与同位于一个Dense Block中的多个层有连接，而与Dense Block外的层是没有连接的。

![2018-12-27_144254](C:\Users\Shelton\Desktop\2018-12-27_144254.png"DenseNet结构")

## 加载预训练网络模型

torchvision是服务于PyTorch框架的，用于进行图片处理和生成一些主流模型的库。使用该库可以方便的加载PyTorch的预训练模型。首先使用pip安装torchvision库：

```python
pip install torchvision
```

创建densenet121模型实例，并加载预训练参数：

```
cnn = torchvision.models.densenet121(pretrained = True) #pretrained =True即为加载预训练参数，默认不加载。
```

冻结所有模型参数，使其值在反向传播中不改变：

```python
for param in cnn.parameters():
    param.requires_grad = False
```

改变模型全连接层输出的个数为256：

```python
num_features = cnn.classifier.in_features
cnn.classifier = nn.Linear(num_features, 256)
```

此处不需要担心新建的全连接层参数会被冻结，因为新建的层参数是默认获取梯度的。

## 训练神经网络

损失函数选择CrossEntropy，优化器选择Adam：

```python
optimizer = Adam(cnn.parameters(), lr=0.001, betas=(0.9, 0.999))  # 选用AdamOptimizer
loss_fn = nn.CrossEntropyLoss()  # 定义损失函数
```

下面是完整的训练过程：

```python
# 训练并评估模型
data = Dataset()
model = Model(data)

best_accuracy = 0
for i in range(args.EPOCHS):
    cnn.train()
    x_train, y_train, x_test, y_test = data.next_batch(args.BATCH)  # 读取数据

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = x_train.float()

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_test = x_test.float()
    
    if cuda_avail:
        x_train = Variable(x_train.cuda())
        y_train = Variable(y_train.cuda())
        x_test = Variable(x_test.cuda())
        y_test = Variable(y_test.cuda())
        
    outputs = cnn(x_train)
    _, prediction = torch.max(outputs.data, 1)
    
    optimizer.zero_grad()

    # calculate the loss according to labels
    loss = loss_fn(outputs, y_train)
    # backward transmit loss
    loss.backward()

    # adjust parameters using Adam
    optimizer.step()

    # 若测试准确率高于当前最高准确率，则保存模型
    train_accuracy = eval(model, x_test, y_test)
    if train_accuracy > best_accuracy:
        best_accuracy = train_accuracy
        model.save_model(cnn, MODEL_PATH, overwrite=True)
        print("step %d, best accuracy %g" % (i, best_accuracy))

    print(str(i) + "/" + str(args.EPOCHS))
```

## 小结

本文主要讲解了DenseNet的网络结构，以及在PyTorch框架下如何加载预训练模型并进行fine-tuning。为了在数据集上获得更高的准确率，读者可尝试取消冻结参数的设置，使得卷积层也参与训练。