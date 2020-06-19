from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from dataloader.dataset import *
from models.densenet import *
from utils.other_utils import *
from torch.utils.tensorboard import SummaryWriter
import torch
from sklearn import metrics

BATCH_SIZE = 16
model_path1 = '/cluster/home/it_stu109/wyj/M3DV2/mylib/checkpoint/90 epoch.tar'
model_path2 = './checkpoint/40 epoch.tar'
EVAL = False
ENSEMBLE =False
test_path = r'/cluster/home/it_stu109/wyj/M3DV2/mylib/dataset'
val_path = r'/cluster/home/it_stu109/wyj/M3DV2/mylib/dataset'

writer = SummaryWriter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if ENSEMBLE == False:
    model =densenet264().cuda()
    checkpoint = torch.load(model_path1)
    model.load_state_dict(checkpoint['model_state_dict'])

    if EVAL:
        test_set = VOXDataset(root=val_path, train=False,crop_size=40, move=3)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False, )
        all_test_score = np.zeros((1, 1))
        all_test_label = np.zeros((1, 1))
        for val_images, val_labels in tqdm(test_loader):

            model.eval()  # set model to evaluation mode
            with torch.no_grad():
                test_scores = model(val_images.cuda())

                if all_test_score.shape[1] == 1:
                    all_test_score = test_scores.cpu().clone().detach().numpy()
                    all_test_label = val_labels.cpu().clone().detach().numpy()
                else:
                    all_test_score = np.concatenate((all_test_score, test_scores.cpu().clone().detach().numpy()), 0)
                    all_test_label = np.concatenate((all_test_label, val_labels.cpu().clone().detach().numpy()), 0)

        test_AUC = metrics.roc_auc_score(all_test_label, all_test_score[:, 1])
        test_accuracy = metrics.accuracy_score(all_test_label, all_test_score.argmax(1))
        print(test_AUC, test_accuracy)
    else:
        generate_test_csv(path=test_path)
        test_set = TestDataset(root=test_path,crop_size=40,move=3)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False, )
        all_test_score = np.zeros((1, 1))
        all_test_label = np.zeros((1, 1))

        for val_images in tqdm(test_loader):

            model.eval()  # set model to evaluation mode
            with torch.no_grad():
                test_scores = model(val_images.cuda())

                if all_test_score.shape[1] == 1:
                    all_test_score = test_scores.cpu().clone().detach().numpy()
                else:
                    all_test_score = np.concatenate((all_test_score, test_scores.cpu().clone().detach().numpy()), 0)

        generate_test_csv(path=test_path, predicted=all_test_score[:, 1])

else:
    model1 =densenet264().cuda()
    checkpoint = torch.load(model_path1)
    model1.load_state_dict(checkpoint['model_state_dict'])
    model2 =densenet264().cuda()
    model2= model2.cuda()
    checkpoint = torch.load(model_path2)
    model2.load_state_dict(checkpoint['model_state_dict'])


    if EVAL:
        test_set = VOXDataset(root=val_path, train=False)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False, )
        all_test_score1 = np.zeros((1, 1))
        all_test_score2 = np.zeros((1, 1))
        all_test_label = np.zeros((1, 1))

        for val_images, val_labels in tqdm(test_loader):
            val_images=val_images.cuda()
            val_labels=val_labels.cuda()
            model1.eval()
            model2.eval()
            with torch.no_grad():
                test_scores1 = model1(val_images)
                test_scores2 = model2(val_images)
                if all_test_score1.shape[1] == 1:
                    all_test_score1 = test_scores1.cpu().cpu().clone().detach().numpy()
                    all_test_score2 = test_scores2.cpu().clone().detach().numpy()
                    all_test_label = val_labels.cpu().clone().detach().numpy()
                else:
                    all_test_score1 = np.concatenate((all_test_score1, test_scores1.cpu().cpu().clone().detach().numpy()), 0)
                    all_test_score2 = np.concatenate((all_test_score2, test_scores2.cpu().cpu().clone().detach().numpy()), 0)
                    all_test_label = np.concatenate((all_test_label, val_labels.cpu().cpu().clone().detach().numpy()), 0)
        all_test_score = (all_test_score1 + all_test_score2) * 0.5
        test_AUC = metrics.roc_auc_score(all_test_label, all_test_score[:, 1])
        test_accuracy = metrics.accuracy_score(all_test_label, all_test_score.argmax(1))
        print(test_AUC, test_accuracy)
    else:
        generate_test_csv(path=test_path)
        test_set = TestDataset(root=test_path, crop_size=40, move=3)
        #test_set = TestDataset(root=test_path)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False, )
        all_test_score1 = np.zeros((1, 1))
        all_test_score2 = np.zeros((1, 1))
        all_test_label = np.zeros((1, 1))

        for test_images in tqdm(test_loader):

            model1.eval()
            model2.eval()
            with torch.no_grad():
                test_scores1 = model1(test_images.cuda())
                test_scores2 = model2(test_images.cuda())
                if all_test_score1.shape[1] == 1:
                    all_test_score1 = test_scores1.cpu().clone().detach().numpy()
                    all_test_score2 = test_scores2.cpu().clone().detach().numpy()
                else:
                    all_test_score1 = np.concatenate((all_test_score1, test_scores1.cpu().clone().detach().numpy()), 0)
                    all_test_score2 = np.concatenate((all_test_score2, test_scores2.cpu().clone().detach().numpy()), 0)
        all_test_score = (all_test_score1 + all_test_score2) * 0.5
        print(all_test_score[:,1])
        generate_test_csv(path=test_path, predicted=all_test_score[:, 1])
writer.close()

"""import pandas as pd

df = pd.read_csv(r'/cluster/home/it_stu109/wyj/M3DV2/mylib/dataset/test.csv')
predicted=df['predicted'].values
names=df['name'].values
predicted=(predicted-predicted.min())/(predicted.max()-predicted.min())
val_frame = pd.DataFrame({'name':names,'predicted':predicted})
val_frame.to_csv(r'/cluster/home/it_stu109/wyj/M3DV2/mylib/dataset/test.csv',index=False)"""