import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.datasets import DatasetFolder
import torchvision
from tqdm.notebook import tqdm as tqdm

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')

train_path = '/home/chisc/workspace/wuzhenrong/train'
val_path = '/home/chisc/workspace/wuzhenrong/validation/'
test_path = '/home/chisc/workspace/wuzhenrong/test/'

# scripted_transforms = torch.jit.script(transforms)
# I don't know how to set normalize value, so I copy pytorch documents sample code.
# min max normalization
## cat 0
## dog 1
### Data Augmentation

train_trans = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomRotation((-30, 30)),
     transforms.Resize((224, 224)), 
     transforms.ToTensor(), 
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# val_trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
val_trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
test_trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

batch_size = 32

train_data = ImageFolder(train_path, transform = train_trans)
val_data = ImageFolder(val_path,transform = test_trans)
test_data = ImageFolder(test_path, transform = test_trans)

# shuffle: Each epoch's training sample are different
# drop_last: If the dataset can't be divided by the batch_size, the last data won't be remove
# num_workers: num_workers is depend on your cpu and your RAM, and num_workers can help you preload the batch data and store in RAM,
# if you have lots of num_workers, your preload speed will be fast, but in the other hand, your cpu will have increasing burden
# pin_memory: If this parameter is True, Dataloader will copy the tensor to CUDA's RAM, before return
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
test_loader = DataLoader(test_data, shuffle = True)
print(train_loader)

images, labels = next(iter(train_loader))
# After Normalize
for i in np.arange(3):
  plt.figure(i)
  plt.imshow(images[i].permute(1, 2, 0))
# plt.show()
# Before Normalize
for i in np.arange(3):
  plt.figure(i)
  # Our data are normalized, in order to watch our origin image, so we need to denormalize our data
  mean = torch.tensor([0.485, 0.456, 0.406])
  std = torch.tensor([0.229, 0.224, 0.225])
  tmp = transforms.Normalize(-mean/std, 1/std)(images[i]) # denormalize
  plt.imshow(tmp.permute(1, 2, 0)) # The data in pytorch is (channel, size, size), and we need to change it to (size, size, channel)
  plt.show()

# 1. Input layer
# 2. Convolutional layer
# 3. ReLU layer
# 4. Pooling layer
# 5. Fully-connected layer
class CatDpg(nn.Module):
  def __init__(self):
    super(CatDpg, self).__init__()
    # input_shape = (3, 224, 224)
    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
    #                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    # in_channels: Input channels
    # out_channels: Output channels
    # kernel_size: Fillter size
    # stride: Each step our Fillter move
    # padding: We want our image can remain origin size
    self.cnn = nn.Sequential(
        ## CNN1
        nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1), # padding = kernel_size / 2
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),## (64, 112, 112)
        ## CNN2
        nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),## (128, 56, 56)
        ## CNN3
        nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),## (256, 28, 28)
        ## CNN4
        nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2),## (512, 14, 14)
        ## CNN5
        nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2)## (512, 7, 7)
    )
    self.fc = nn.Sequential(
        nn.Linear(512 * 7 * 7, 1024), # Fully-connected layer
        nn.Dropout(0.4), # Avoid overfitting
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(1024, 2)
    )
  # forward propagation
  def forward(self, x):
    x = self.cnn(x)
    x = x.flatten(1)
    x = self.fc(x)
    return x

device = "cuda" if train_on_gpu else "cpu"
model = CatDpg()

model = model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_func = nn.CrossEntropyLoss()

n_epochs = 30
train_loss_record = []
train_acc_record = []
val_loss_record = []
val_acc_record = []

for epoch in range(n_epochs):
  train_loss = 0.0
  val_loss = 0.0
  train_acc = 0.0
  val_acc = 0.0
  model.train()

  for x, y in tqdm(train_loader):
    x, y = x.to(device), y.to(device)
    prediction = model(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = ((prediction.argmax(dim = 1) == y).float().mean())
    train_acc += acc/len(train_loader)
    train_loss += loss/len(train_loader)

  print(f"[ Train | {epoch+1}/{n_epochs} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
  train_loss_record.append(train_loss)
  train_acc_record.append(train_acc)
#   with torch.no_grad():
  for x, y in tqdm(val_loader):
      x, y = x.to(device), y.to(device)
      prediction = model(x)
      loss = loss_func(prediction, y)
      loss.backward()
      acc = ((prediction.argmax(dim = 1) == y).float().mean())
      val_acc += acc/len(val_loader)
      val_loss += loss/len(val_loader)
  print(f"[ Validation | {epoch+1}/{n_epochs} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")
  val_loss_record.append(val_loss)
  val_acc_record.append(val_acc)
torch.save(model, 'catvsdog.pkl')

plt.figure(1)
plt.title('Training and Validation Loss')
train_l, = plt.plot(train_loss_record, color = 'red')
val_l, = plt.plot(val_loss_record, color = 'blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(handles = [train_l, val_l], labels = ['Training', 'Validation'], loc = 'best')
plt.show()

plt.figure(2)
plt.title('Training and Validation Accuracy')
train_a, = plt.plot(train_acc_record, color = 'red')
val_a, = plt.plot(val_acc_record, color = 'blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(handles = [train_a, val_a], labels = ['Training', 'Validation'], loc = 'best')
plt.show()

i = 0
for x, y in test_loader:
  i += 1
  if train_on_gpu:
    x, y = x.cuda(), y.cuda()
  output = model(x)
  out = output.argmax(dim = 1)
  out = out.to('cpu').numpy()
  # print(out)
  if i % 10 == 0:
    plt.figure(i)
    if out[0] == 0:
      plt.title('Predict: cat')
    else:
      plt.title('Predict: dog')
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    x = x.squeeze()
    tmp = transforms.Normalize(-mean/std, 1/std)(x) # denormalize
    tmp = tmp.to('cpu')
    plt.imshow(tmp.permute(1, 2, 0))
    plt.show()
