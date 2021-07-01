#-------------------------------------
# Project: Few-shot Learning Based on Hierarchical Classification via Multi-granularity Relation Networks
# Date: 2021.06.29
# Author: Yuling Su
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import scipy.io as sciio
import scipy as sp
import scipy.stats
import math
import argparse
import random
import math
from resnet import resnet12,RR


parser = argparse.ArgumentParser(description="Five Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 512)
parser.add_argument("-p","--coarse_relation_dim",type = int, default = 128)  
parser.add_argument("-r","--fine_relation_dim",type = int, default = 128)  
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 10)  
parser.add_argument("-e","--episode",type = int, default= 150000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.1)
parser.add_argument("-g","--gpu",type=int, default=1)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()


FEATURE_DIM = args.feature_dim
FINE_RELATION_DIM = args.fine_relation_dim
coarse_RELATION_DIM = args.coarse_relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS=args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a),scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
   
        return out 

class RelationNetwork(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))  
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def get_new_sample_labels(labels):
    new_sample_labels = []
    for i in range(CLASS_NUM):
        idx = i*5
        new_sample_labels.append(labels[idx].item())
    return new_sample_labels

def get_new_labels(labels,depend_labels): 
    
    new_labels = []
    for c in labels:
        label = []
        for j in range(len(depend_labels)):
            if c == depend_labels[j]:
                label.append(1)
            else:
                label.append(0)
        new_labels.append(label)
    return new_labels

    new_coarses_labels = [depend_labels[i] for i in labels]

    return new_coarses_labels

def get_predict_labels(labels,depend_labels):  

    predict_labels = [depend_labels[i] for i in labels]

    return predict_labels

def get_coarses_labels(labels,coarse_LABEL):  
    coarses_labels = [coarse_LABEL[i]  for i in labels]

    return coarses_labels



def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders() 

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = resnet12()
    coarse_relation_network = RR(coarse_RELATION_DIM)
    fine_relation_network = RR(FINE_RELATION_DIM)
    mse = nn.MSELoss().cuda(GPU)

    feature_encoder.apply(weights_init)   
    coarse_relation_network.apply(weights_init)
    fine_relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    coarse_relation_network.cuda(GPU)
    fine_relation_network.cuda(GPU)
    
    feature_encoder_optim = torch.optim.SGD(feature_encoder.parameters(),momentum=0.9,lr=LEARNING_RATE)  
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)  
    coarse_relation_network_optim = torch.optim.SGD(coarse_relation_network.parameters(),momentum=0.9,lr=LEARNING_RATE)
    coarse_relation_network_scheduler = StepLR(coarse_relation_network_optim,step_size=100000,gamma=0.5)
    fine_relation_network_optim = torch.optim.SGD(fine_relation_network.parameters(),momentum=0.9,lr=LEARNING_RATE)
    fine_relation_network_scheduler = StepLR(fine_relation_network_optim,step_size=100000,gamma=0.5)     

    if os.path.exists(str("./Omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./Omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    
    if os.path.exists(str("./Omniglot_fine_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        coarse_relation_network.load_state_dict(torch.load(str("./Omniglot_fine_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load coarse relation network success")
    
    if os.path.exists(str("/home/zh510/文档/我的实验/Relation Network/LearningToCompare_FSL/Few-shot/relation-voc-测试结果/5-way-1-shot-11.22/fine_relation_encoder/training50000.pkl")):
        fine_relation_network.load_state_dict(torch.load(str("/home/zh510/文档/我的实验/Relation Network/LearningToCompare_FSL/Few-shot/relation-voc-测试结果/5-way-1-shot-11.22/fine_relation_encoder/training50000.pkl")))
        print("load fine relation network success")
 
    print("Training...")

    last_accuracy = 0.0
    last_episode = 0
    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode) 
        coarse_relation_network_scheduler.step(episode)
        fine_relation_network_scheduler.step(episode)

    
        degrees = random.choice([0,90,180,270])
        task = tg.OmniglotTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
        batch_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)

        samples,sample_coarse_labels,sample_labels = sample_dataloader.__iter__().next()
        batches,batch_coarse_labels,batch_labels = batch_dataloader.__iter__().next()

        sample_features = feature_encoder(Variable(samples).cuda(GPU))
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,2,2)
        sample_features = torch.mean(sample_features,1).unsqueeze(0).cuda(GPU)
        batch_features = feature_encoder(Variable(batches).cuda(GPU))
     
        sample_features_ext = sample_features.repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,2,2)
        fine_relations = fine_relation_network(relation_pairs).view(-1,CLASS_NUM)
        fine_relations = F.softmax(fine_relations,dim=1)
       
        _,fine_predcit = torch.max(fine_relations.data,1)
        fine_reward =[1 if fine_predcit[j]==batch_labels[j] else 0 for j in range(batch_labels.size(0))]
        fine_accuracy = np.sum(fine_reward)/1.0/BATCH_NUM_PER_CLASS/CLASS_NUM

        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda(GPU)
        fine_loss = mse(fine_relations,one_hot_labels)


        sample_labels = torch.Tensor(get_new_sample_labels(sample_labels)).view(-1,1).int()
        sample_coarse_labels = torch.Tensor(get_new_sample_labels(sample_coarse_labels)).view(-1,1).int()
        sample_coarses = {} 
        for i in range(sample_labels.size(0)):
            if sample_coarse_labels[i].item() not in sample_coarses.keys():
                sample_coarses[sample_coarse_labels[i].item()] = [i]
            else:
                sample_coarses[sample_coarse_labels[i].item()].append(i)
        batch_coarses_num = len(sample_coarses.keys())
    
        for i, (key,value) in enumerate(sample_coarses.items()):
            sample_coarses_features_ext = sample_features[0,value,:] 
            sample_coarses_features_ext = sample_coarses_features_ext.unsqueeze(0)
            sample_coarses_features_ext = torch.mean(sample_coarses_features_ext,1)  

            if i == 0:
                sample_coarses_features_exts = sample_coarses_features_ext
            else:
                sample_coarses_features_exts = torch.cat((sample_coarses_features_exts,sample_coarses_features_ext),0) 
        sample_coarses_features_exts = sample_coarses_features_exts.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)  
        batch_features_exts = batch_features.unsqueeze(0).repeat(batch_coarses_num,1,1,1,1)  
        batch_features_exts = torch.transpose(batch_features_exts,0,1)
        coarse_relation_pairs = torch.cat((sample_coarses_features_exts,batch_features_exts),2) 
        coarse_relation_pairs = coarse_relation_pairs.view(-1,FEATURE_DIM*2,2,2)
        coarses_relations = coarse_relation_network(coarse_relation_pairs).view(-1,batch_coarses_num)
        coarses_relations = F.softmax(coarses_relations,dim=1)
        new_batch_coarses_labels = torch.Tensor(get_new_labels(batch_coarse_labels,list(sample_coarses.keys()))).view(-1,batch_coarses_num).cuda(GPU)
        coarse_loss = mse(coarses_relations,new_batch_coarses_labels)  
        
        _,predict_coarses_labels = torch.max(coarses_relations.data,1)  
        batch_predict_coarses_labels = torch.Tensor(get_predict_labels(predict_coarses_labels,list(sample_coarses.keys()))).view(-1,1)
        
        coarses_reward = [1 if batch_predict_coarses_labels[j]==batch_coarse_labels[j] else 0 for j in range(len(batch_coarse_labels))]   
        coarse_accuracy = np.sum(coarses_reward)/1.0/CLASS_NUM/BATCH_NUM_PER_CLASS  

        relations = torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).cuda(GPU)
        for i,(key,value) in enumerate(sample_coarses.items()):
            for j in value:
                relations[:,j] = 0.5*(0.5*fine_relations[:,j] + 0.5*coarses_relations[:,i])

        _,predcit = torch.max(relations.data,1)
        reward =[1 if predcit[j]==batch_labels[j] else 0 for j in range(batch_labels.size(0))]
        train_accuracy = np.sum(reward)/1.0/BATCH_NUM_PER_CLASS/CLASS_NUM

        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda(GPU)
        all_loss = mse(relations,one_hot_labels)
        
        loss = all_loss+0.2*(fine_loss + coarse_loss)

        if (episode+1)%100 == 0:
                print("train:%6d, fine_loss:%.6f, coarse_loss:%.6f, loss:%.6f, fine_accuracy:%.4f, coarse_accuracy:%.4f, train_accuracy:%.4f" %(episode+1,fine_loss,coarse_loss,loss,fine_accuracy,coarse_accuracy,train_accuracy))

        feature_encoder.zero_grad()
        fine_relation_network.zero_grad()
        coarse_relation_network.zero_grad()
          
        loss.backward()

        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(),0.5)  
        torch.nn.utils.clip_grad_norm_(fine_relation_network.parameters(),0.5)
        torch.nn.utils.clip_grad_norm_(coarse_relation_network.parameters(),0.5)
        

        feature_encoder_optim.step()  #
        fine_relation_network_optim.step()
        coarse_relation_network_optim.step()
        


        if (episode+1)%5000 == 0:

            # test
            print("Testing...")
            accuracies = []
            fine_accuracies = []
            coarse_accuracies = []
            for test_episode in range(TEST_EPISODE):
                degrees = random.choice([0,90,180,270])
                task = tg.OmniglotTask(metatest_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS,)
                sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
                test_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)

                sample_images,sample_coarse_labels,sample_labels = sample_dataloader.__iter__().next()
                test_images,test_coarse_labels,test_labels = test_dataloader.__iter__().next()
                
                sample_labels = torch.Tensor(get_new_sample_labels(sample_labels)).view(-1,1).int()
                sample_coarse_labels = torch.Tensor(get_new_sample_labels(sample_coarse_labels)).view(-1,1).int()

                sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) 
                # print(sample_features.size())
                sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,2,2)
                sample_features = torch.mean(sample_features,1).unsqueeze(0).cuda(GPU) 
                test_features = feature_encoder(Variable(test_images).cuda(GPU)) 

                sample_features_ext = sample_features.repeat(CLASS_NUM*BATCH_NUM_PER_CLASS,1,1,1,1)  
                test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)  
                test_features_ext = torch.transpose(test_features_ext,0,1)
                relation_pairs = torch.cat((sample_features_ext,test_features_ext),2) 
                relation_pairs = relation_pairs.view(-1,FEATURE_DIM*2,2,2)

                fine_relations = fine_relation_network(relation_pairs).view(-1,CLASS_NUM)
                fine_relations = F.softmax(fine_relations,dim=1)

                _,fine_predcit = torch.max(fine_relations.data,1)
                fine_reward =[1 if fine_predcit[j]==test_labels[j] else 0 for j in range(test_labels.size(0))]
                fine_accuracy = np.sum(fine_reward)/1.0/BATCH_NUM_PER_CLASS/CLASS_NUM
                fine_accuracies.append(fine_accuracy)

                sample_coarses = {}  
                for i in range(sample_labels.size(0)):
                    if sample_coarse_labels[i].item() not in sample_coarses.keys():
                        sample_coarses[sample_coarse_labels[i].item()] = [i]
                    else:
                        sample_coarses[sample_coarse_labels[i].item()].append(i)
                test_coarses_num = len(sample_coarses.keys())
                
                for i, (key,value) in enumerate(sample_coarses.items()):
                    
                    sample_coarses_features_ext = sample_features[0,value,:] 
                    sample_coarses_features_ext = sample_coarses_features_ext.unsqueeze(0)
                    sample_coarses_features_ext = torch.mean(sample_coarses_features_ext,1)  

                    if i == 0:
                        sample_coarses_features_exts = sample_coarses_features_ext
                    else:
                        sample_coarses_features_exts = torch.cat((sample_coarses_features_exts,sample_coarses_features_ext),0) 
                sample_coarses_features_exts = sample_coarses_features_exts.unsqueeze(0).repeat(CLASS_NUM*BATCH_NUM_PER_CLASS,1,1,1,1)  
                test_features_exts = test_features.unsqueeze(0).repeat(test_coarses_num,1,1,1,1) 
                test_features_exts = torch.transpose(test_features_exts,0,1)  

                coarse_relation_pairs = torch.cat((sample_coarses_features_exts,test_features_exts),2)
                coarse_relation_pairs = coarse_relation_pairs.view(-1,FEATURE_DIM*2,2,2)

                coarses_relations = coarse_relation_network(coarse_relation_pairs).view(-1,test_coarses_num)
                coarses_relations = F.softmax(coarses_relations,dim=1)
                
                _,predict_coarses_labels = torch.max(coarses_relations.data,1)  
                test_predict_coarses_labels = torch.Tensor(get_predict_labels(predict_coarses_labels,list(sample_coarses.keys()))).view(-1,1)
                coarses_reward = [1 if test_predict_coarses_labels[j]==test_coarse_labels[j] else 0 for j in range(len(test_coarse_labels))]   
                coarse_accuracy = np.sum(coarses_reward)/1.0/CLASS_NUM/BATCH_NUM_PER_CLASS 
                coarse_accuracies.append(coarse_accuracy)
            
                relations = torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).cuda(GPU)
                for i,(key,value) in enumerate(sample_coarses.items()):
                    for j in value:
                        relations[:,j] = 0.5*(0.6*fine_relations[:,j] + 0.4*coarses_relations[:,i])

                _,predcit = torch.max(relations.data,1)
                reward =[1 if predcit[j]==test_labels[j] else 0 for j in range(test_labels.size(0))]
                test_accuracy = np.sum(reward)/1.0/BATCH_NUM_PER_CLASS/CLASS_NUM
                accuracies.append(test_accuracy)

                if (test_episode+1)%100 == 0:
                        print("test:%6d, fine_accuracy:%.4f, coarse_accuracy:%.4f, test_accuracy:%.4f" %(test_episode+1,fine_accuracy,coarse_accuracy,test_accuracy))

            test_accuracy,h = mean_confidence_interval(accuracies)
            coarse_accuracy,coarse_h = mean_confidence_interval(coarse_accuracies)
            fine_accuracy,fine_h = mean_confidence_interval(fine_accuracies)

            print("test: fine_accuracy:%.4f, fine_h:%.4f, coarse_accuracy:%.4f, coarse_h:%.4f, test_accuracy:%.4f, test_h:%.4f, last_episode:%6d, last_accuracy:%.4f" %(fine_accuracy,fine_h,coarse_accuracy,coarse_h,test_accuracy,h,last_episode,last_accuracy))

            if test_accuracy > last_accuracy:

              
                torch.save(feature_encoder.state_dict(),str("./Omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                torch.save(coarse_relation_network.state_dict(),str("./Omniglot_coarse_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                torch.save(fine_relation_network.state_dict(),str("./Omniglot_fine_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
       

                print("save networks for episode:",episode)

                last_accuracy = test_accuracy
                last_episode = episode + 1
    
    print("best episode: ",last_episode,"best accuracy: ",last_accuracy,"h: ",h)



if __name__ == '__main__':
    main()