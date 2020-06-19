import torch
from torch import  optim
from tqdm import tqdm
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from dataloader.dataset import *
from models.densenet  import *
from utils.other_utils import *
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set=VOXDataset(root=r'/cluster/home/it_stu109/wyj/M3DV2/mylib/dataset',train=True,crop_size=40, move=3)
test_set=VOXDataset(root=r'/cluster/home/it_stu109/wyj/M3DV2/mylib/dataset',train=False,crop_size=40)

LR=0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 100
STEPS=2*NUM_EPOCHS
SAVE_PATH='./checkpoint'
mixrate=0.4

train_loader1 = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False,)
train_loader2 = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False,)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False,)

writer = SummaryWriter()

model=densenet264().cuda()
optimizer=optim.Adam(params=model.parameters(),lr=LR)
scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)

model.train()
for epoch in range(NUM_EPOCHS):

    correct=0
    num_data=0
    all_score = np.zeros((1, 1))
    all_label = np.zeros((1, 1))
    for i, data in enumerate(zip(train_loader1, train_loader2)):

        images=data[0][0]
        labels=data[0][1]
        images_1=data[1][0]
        labels_1=data[1][1]


        mixedimages=mixrate*images+(1-mixrate)*images_1
        mixedimages=mixedimages.cuda()
        labels=labels.cuda()
        labels_1=labels_1.cuda()





        scores = model(mixedimages)


        loss =F.cross_entropy(scores, labels)*mixrate+F.cross_entropy(scores, labels_1)*(1-mixrate)
        optimizer.zero_grad()
        loss.backward()
        print(loss)
        optimizer.step()


    all_test_score = np.zeros((1, 1))
    all_test_label = np.zeros((1, 1))
    model.eval()
    for val_images, val_labels in iter(test_loader):
        val_images = val_images.cuda()
        val_labels = val_labels.cuda()
        with torch.no_grad():
            test_scores = model(val_images)

            if all_test_score.shape[1] == 1:
                all_test_score = test_scores.cpu().clone().detach().numpy()
                all_test_label = val_labels.cpu().clone().detach().numpy()
            else:
                all_test_score = np.concatenate((all_test_score, test_scores.cpu().clone().detach().numpy()), 0)
                all_test_label = np.concatenate((all_test_label, val_labels.cpu().clone().detach().numpy()), 0)

    test_AUC = metrics.roc_auc_score(all_test_label, all_test_score[:, 1])

    scheduler.step()

    print(test_AUC,epoch)

    writer.add_scalar('Loss/train', loss, epoch)


    if epoch%5 == 0 or epoch ==NUM_EPOCHS-1:

        checkpath=os.path.join(SAVE_PATH, '%s epoch.tar' % epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpath)
writer.close()

all_test_score = np.zeros((1, 1))
all_test_label = np.zeros((1, 1))

for val_images, val_labels in tqdm(test_loader):
    val_images=val_images.cuda()
    val_labels=val_labels.cuda()
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        test_scores = model(val_images)

        if all_test_score.shape[1] == 1:
            all_test_score = test_scores.cpu().clone().detach().numpy()
            all_test_label = val_labels.cpu().clone().detach().numpy()
        else:
            all_test_score =np.concatenate((all_test_score, test_scores.cpu().clone().detach().numpy()), 0)
            all_test_label =np.concatenate((all_test_label, val_labels.cpu().clone().detach().numpy()), 0)

test_AUC = metrics.roc_auc_score(all_test_label, all_test_score[:,1])
test_accuracy = metrics.accuracy_score(all_test_label,all_test_score.argmax(1))
print(test_AUC, test_accuracy)