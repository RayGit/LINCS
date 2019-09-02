import os
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix,accuracy_score
from tqdm import tqdm
from losses import CenterLoss
from tensorboardX import SummaryWriter
import numpy as np

class Trainer(object):

    #cuda = torch.cuda.is_available()
    #torch.backends.cudnn.benchmark = True
    def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=1):
        self.model = model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = torch.nn.DataParallel(self.model,device_ids=range(torch.cuda.device_count()))
        self.model.to(device)
        #self.model.load_state_dict(torch.load("checkpoints/exp1/ model_370.pkl")['weight'])
        self.optimizer = optimizer
        self.loss_1 = loss_f().cuda()
        self.loss_2 = CenterLoss().cuda()
        self.optimizer_center=torch.optim.Adam(params=self.loss_2.parameters())
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.writer = SummaryWriter()


    def _iteration(self, data_loader, ep ,is_train=True):
        loop_loss = []
        #loop_loss_a = []
        outputlabel= []
        targetlabel = []
        for img1,target in tqdm(data_loader):

            img1,target = img1.cuda(),target.cuda()
            target = target.squeeze_()
            out1,out2= self.model(img1)
            print(out2.size())
            loss_1 = self.loss_1(out2,target)
            loss_2 = self.loss_2(out1,target)
            loss_step = loss_1.data.item()
            print(">>>loss:", loss_step)
            loop_loss.append(loss_1.data.item() / len(data_loader))
            #loop_loss_a.append(loss_a.data.item() / len(data_loader))
            #accuracy.append((output.data.max(1)[1] == target.data).sum().item())
            if is_train:
                self.optimizer.zero_grad()
                self.optimizer_center.zero_grad()
                loss_1.backward(retain_graph=True)
                loss_2.backward(retain_graph=True)
                self.optimizer_center.step()
                self.optimizer.step()

            output = F.softmax(out2, dim=1)
            output = output.cpu()
            output = output.data.numpy()
            output = np.argmax(output, axis=1)
            target = target.cpu().data.numpy()
            # target = np.argmax(target,axis=1)
            target = np.reshape(target, [-1])
            output = np.reshape(output, [-1])
            target = target.astype(np.int8)
            output = output.astype(np.int8)
            outputlabel.append(output)
            targetlabel.append(target)

        if is_train:
            self.writer.add_scalar('train/loss_epoch', sum(loop_loss), ep)
            targetlabel = np.reshape(np.array(targetlabel), [-1]).astype(np.int)
            outputlabel = np.reshape(np.array(outputlabel), [-1]).astype(np.int)
            accuracy = accuracy_score(targetlabel, outputlabel)
            self.writer.add_scalar('train/accuracy', accuracy, ep)
            #self.writer.add_scalar('train/loss_a_epoch', sum(loop_loss_a), ep)
            #self.writer.add_scalar('train/accuracy',sum(accuracy)/len(data_loader.dataset),ep)
        else:
            print(targetlabel)
            print(outputlabel)
            targetlabel = np.reshape(np.array(targetlabel),[-1]).astype(np.int)
            outputlabel = np.reshape(np.array(outputlabel),[-1]).astype(np.int)
            accuracy = accuracy_score(targetlabel, outputlabel)
            print(accuracy)
            matrixs = confusion_matrix(targetlabel,outputlabel)
            np.save('matrixs/matrixs_'+ str(ep) + '.npy',matrixs)

            self.writer.add_scalar('test/accuracy', accuracy, ep)
            self.writer.add_scalar('test/loss_epoch', sum(loop_loss), ep)
            #self.writer.add_scalar('test/loss_a_epoch', sum(loop_loss_a), ep)
            #self.writer.add_scalar('test/accuracy',sum(accuracy)/len(data_loader.dataset),ep)
        mode = "train" if is_train else "test"
        #print(">>>[{mode}] loss: {loss}/accuracy: {accuracy}".format(mode=mode,loss=sum(loop_loss),accuracy=sum(accuracy)/len(data_loader.dataset)))
        print(">>>[{mode}] loss: {loss}".format(mode=mode,loss=sum(loop_loss)))
        return loop_loss

    def train(self, data_loader,ep):
        self.model.train()
        with torch.enable_grad():
            loss = self._iteration(data_loader,ep)
            #pass

    def test(self, data_loader,ep):
        self.model.eval()
        with torch.no_grad():
            loss = self._iteration(data_loader,ep,is_train=False)

    def loop(self, epochs, train_data, test_data, scheduler=None):
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print("epochs: {}".format(ep))
            self.train(train_data,ep)
            if (ep % self.save_freq == 0):
                self.save(ep)
            self.test(test_data,ep)


    def save(self, epoch, **kwargs):
        model_out_path = self.save_dir
        state = {"epoch": epoch, "weight": self.model.state_dict()}
        if not os.path.exists(model_out_path):
            os.makedirs(model_out_path)
        torch.save(state, model_out_path + '/ model_{epoch}.pkl'.format(epoch=epoch))