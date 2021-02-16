import os
import torch
import models
import pandas as pd
import torch.nn as nn

h = 6
w = 6

model = models.GroundTruth0()

if os.path.exists("base/groundtruth.pth"):
    model.load_state_dict(torch.load("base/groundtruth.pth"))
else:
    os.mkdir("base")
    torch.save(model.state_dict(),"base/groundtruth.pth")

if not os.path.exists("datasets/data"):
    os.mkdir("datasets/data")

#model.cuda()
print("n params:")
print(sum([len(w.view(-1)) for w in model.parameters()]))
#for parameter in model.parameters():
#    print(parameter.size())


model.eval()


data = {"image":[], "category":[]}

for i in range(400):

    #fake_img = torch.randn([1, 3, h, w], dtype=torch.float32)
    fake_img = torch.empty(1,3,h,w).normal_(mean=128,std=32)
    #fake_img = fake_img.to('cuda')
    #print(fake_img)
    output = model(fake_img)
    #_, predicted = torch.max(output, 1)
    data["image"].append(fake_img[0].detach())
    data["category"].append(output[0].detach())
    #data["category"].append(predicted.item())

torch.save(data, "datasets/data/train.pt")

data = {"image":[], "category":[]}

for i in range(100):

    #fake_img = torch.randn([1, 3, h, w], dtype=torch.float32)
    fake_img = torch.empty(1,3,h,w).normal_(mean=128,std=32)
    #fake_img = fake_img.to('cuda')
    output = model(fake_img)
    #_, predicted = torch.max(output, 1)
    data["image"].append(fake_img[0].detach())
    data["category"].append(output[0].detach())
    #data["category"].append(predicted.item())

torch.save(data, "datasets/data/test.pt")


