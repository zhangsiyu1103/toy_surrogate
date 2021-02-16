import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
#import options as opt
import os
import time
import sys
import numpy as np
from torch.autograd import Variable

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

def init_model(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)
    return model



def test_model_regression(model, dataset):
    model.eval()
    loader = None
    loss_total = 0
    batch_cnt = 0
    criterion = nn.MSELoss()
    if hasattr(dataset, 'test_loader'):
        loader = dataset.test_loader
    elif hasattr(dataset, 'val_loader'):
        loader = dataset.val_loader
    else:
        raise NotImplementedError('Unknown dataset!')
    train_loader = dataset.train_loader
    train_loss_total = 0
    train_batch_cnt = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to("cuda")
            targets = targets.to("cuda")
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            loss_total+=loss.item()
            batch_cnt += 1
            #_, predicted = outputs.max(1)
            #total += targets.size(0)
            #correct += predicted.eq(targets).sum().item()
    #loss_avg = loss_total/batch_cnt
    #print("test loss:", loss_avg)
    #acc = 100.0 * correct / total
    return loss_total/batch_cnt





def train_regression(model, dataset, save_path, surrogate = False, teacher = None,
                        epochs=80, lr=0.01,
                        momentum=0.09,
                        weight_decay=5e-4):
    model.cuda()
    loss_best = sys.maxsize
    model_best = None
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30,
                                              gamma=0.1)

    for i in range(1, epochs + 1):
        print('epoch',i)
        model.train()
        loss_total = 0
        batch_cnt = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(dataset.train_loader):
            inputs = inputs.to("cuda")
            targets = targets.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            batch_cnt += 1
        scheduler.step()
        test_loss = test_model_regression(model, dataset)
        print('train loss: ', loss_total/batch_cnt)
        print('test loss: ',test_loss)
        if test_loss < loss_best:
            loss_best = test_loss
            model.loss = test_loss
            model_best = model
            torch.save(model_best, save_path)
        if surrogate:
            diff_s,diff_o = surrogate_model(model,teacher,dataset.train_loader_s,dataset.test_loader_s)
            print("avg prediction error; avg original error")
            print(diff_s,diff_o)
    return model_best, loss_best


def surrogate_model(model,teacher, train_queue, valid_queue):
  teacher.cuda()
  all_input_grad = []
  r = None
  f = None
  length = 0
  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    train_out = model(input)
    cur_r = train_out-target
    if r == None:
      r = cur_r.unsqueeze(0)
    else:
      r = torch.cat([r,cur_r.unsqueeze(0)])
    input_grad = _concat(torch.autograd.grad(train_out, model.parameters())).data.cpu()
    all_input_grad.append(input_grad)


  all_diff_s = None
  all_diff_o = None
  for step, (input, target) in enumerate(valid_queue):

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    valid_out = model(input)
    valid_grad = _concat(torch.autograd.grad(valid_out, model.parameters())).data
    g = torch.tensor([torch.dot(x.T.cuda(), valid_grad) for x in all_input_grad])

    length = len(all_input_grad)
    H = torch.randn(length,length).cuda()
    for i in range(length):
      for j in range(length):
        H[i,j] = torch.dot(all_input_grad[i].T.cuda(),all_input_grad[j].cuda())
    ret = valid_out-torch.tensordot(torch.tensordot(g.cuda(),torch.inverse(H), dims =1),r.cuda(),dims=1)
    all_train = teacher(input)
    diff_s = torch.abs(all_train-ret)
    diff_o = torch.abs(all_train-valid_out)
    print("surrogate diff;original diff")
    print(diff_s,diff_o)
    if all_diff_s is None:
        all_diff_s = diff_s
    else:
        all_diff_s = torch.cat([all_diff_s,diff_s])
    if all_diff_o is None:
        all_diff_o = diff_o
    else:
        all_diff_o = torch.cat([all_diff_o,diff_o])

  return all_diff_s.mean(), all_diff_o.mean()
