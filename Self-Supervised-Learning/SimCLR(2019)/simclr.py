import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)



run_info = '-'

if not os.path.exists('checkpoints/{}'.format(run_info)):
    os.mkdir('checkpoints/{}'.format(run_info))

log = logging.getLogger('staining_log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fileHandler = logging.FileHandler('checkpoints/{}/log.txt'.format(run_info))
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
#
log.addHandler(fileHandler)
log.addHandler(streamHandler)




class SimCLR(object):

    def __init__(self, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        
        self.device = 'cuda'
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(256) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / 0.07
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        total_loss = []
        total_acc1 = []
        total_acc5 = []
        
        for epoch_counter in range(self.args.epochs):
            for i, (images, _) in enumerate(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                total_loss += [loss.item()]
                total_acc1 += [top1[0]]
                total_acc5 += [top5[0]]
                

                if n_iter % self.args.logeverynsteps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                
                if i % 100 == 0:
                    
                    log.info("Epoch : {}/{}, Batch : {}/{}, loss : {}, acc_top1 : {}, acc_top5 : {}"
                             .format(epoch_counter+1, self.args.epochs, i+1, len(train_loader), 
                                    sum(total_loss) / len(total_loss),
                                    sum(total_acc1) / len(total_acc1),
                                    sum(total_acc5) / len(total_acc5)))
                    
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    log.info('batch : {}/{}'.format(i+1, len(train_loader)))
                    log.info('loss : {}'.format(sum(total_loss) / len(total_loss)))
                    log.info('acc/top1 : {}'.format(sum(total_acc1) / len(total_acc1)))
                    log.info('acc/top5 : {}'.format(sum(total_acc5) / len(total_acc5)))
                    log.info('\n')
                    
                n_iter += 1
                
                
            self.scheduler.step()
            
            log.info('\n')
            log.info("Epoch : {}/{}, Batch : {}/{}, loss : {}, acc_top1 : {}, acc_top5 : {}"
                                 .format(epoch_counter+1, self.args.epochs, i+1, len(train_loader), 
                                        sum(total_loss) / len(total_loss),
                                        sum(total_acc1) / len(total_acc1),
                                        sum(total_acc5) / len(total_acc5)))
        
            torch.save(self.model.state_dict(), 'checkpoints/{}/Epoch {}, Loss {} acc_top1 {} acc_top5 {}'
                       .format(epoch_counter+1, sum(total_loss) / len(total_loss),
                                                sum(total_acc1) / len(total_acc1),
                                                sum(total_acc5) / len(total_acc5)))
            
            log.info('\n')
            
            
            
        #log.info("Training has finished.")
        # save model checkpoints
        #checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        #save_checkpoint({
        #    'epoch': self.args.epochs,
        #    'arch': self.args.arch,
        #    'state_dict': self.model.state_dict(),
        #    'optimizer': self.optimizer.state_dict(),
        #}, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        #logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
