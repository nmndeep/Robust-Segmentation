import torch 
import argparse, os
import yaml
import time
import datetime
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler, SequentialSampler
import torch.utils.data as data
from torchvision import transforms

from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer, create_optimizers, adjust_learning_rate
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, Logger, makedir
from val import evaluate
# from mmcv.utils import Config
# from mmcv.runner import get_dist_info
from semseg.datasets.dataset_wrappers import *



class Trainer:

    def __init__(self, gpu, cfg):

        self.gpu = gpu

        self.train_cfg, self.eval_cfg = cfg['TRAIN'], cfg['EVAL']
        self.dataset_cfg, self.model_cfg = cfg['DATASET'], cfg['MODEL']
        self.loss_cfg, self.optim_cfg, self.sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
        self.epochs, self.lr = self.train_cfg['EPOCHS'], self.optim_cfg['LR']
        self.bs = self.train_cfg['BATCH_SIZE']

        self.world_size = torch.cuda.device_count()
        if self.train_cfg['DDP']:
            self.setup_distributed(world_size=self.world_size)


        self.ignore_label = -1

        # self.model = eval(self.model_cfg['NAME'])(self.model_cfg['BACKBONE'], self.dataset_cfg['N_CLS'])
        self.model = eval(self.model_cfg['NAME'])(self.model_cfg['BACKBONE'], self.dataset_cfg['N_CLS'], self.model_cfg['PRETRAINED'])
        self.model = self.model.to(self.gpu)


        self.save_dir = str(cfg['SAVE_DIR'])

      
        
        if self.gpu == 0:
            self.save_path = f'{self.save_dir}/standard_logs/' + str(self.dataset_cfg['NAME'])  + "/" + str(self.model_cfg['NAME']) + '_' + str(self.model_cfg['BACKBONE']) +f'_{str(datetime.datetime.now())[:-7].replace(" ", "-").replace(":", "_")}_' +str(cfg['ADDENDUM'])
            makedir(self.save_path)
            makedir(self.save_path +"/results")

            self.logger = Logger(self.save_path + "/train_log")

        self.train_loader, self.val_loader = self.dataloaders()

        if self.gpu == 0:
            print("No. of GPUS:", torch.cuda.device_count())
            self.logger.log(str(cfg))
            self.logger.log(str(self.model))

        self.init_optim_log()

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu]) #, find_unused_parameters=True)


    

    def setup_distributed(self, address='localhost', port='12354', world_size=6):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        torch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def init_optim_log(self): 

        model = self.model

        self.loss_fn = get_loss(self.loss_cfg['NAME'], self.ignore_label, None)
        # self.loss_fn = torch.nn.NLLLoss(ignore_index=-1)
        self.optimizer = get_optimizer(model, self.optim_cfg['NAME'], self.lr, self.optim_cfg['WEIGHT_DECAY'])
        # self.optimizer = create_optimizers(model, self.lr, self.optim_cfg['WEIGHT_DECAY'], self.gpu)
        self.scheduler = get_scheduler(self.sched_cfg['NAME'], self.optimizer, self.epochs * self.iters_per_epoch, self.sched_cfg['POWER'], self.iters_per_epoch * self.sched_cfg['WARMUP'], self.sched_cfg['WARMUP_RATIO'])
        # self.scheduler2 = get_scheduler(self.sched_cfg['NAME'], self.optimizer[1], self.epochs * self.iters_per_epoch, self.sched_cfg['POWER'], self.iters_per_epoch * self.sched_cfg['WARMUP'], self.sched_cfg['WARMUP_RATIO'])

        self.scaler = GradScaler(enabled=self.train_cfg['AMP'])
        # self.writer = SummaryWriter(self.save_path + "/results")


    def dataloaders(self):

        
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': self.train_cfg['BASE_SIZE'], 'crop_size': self.train_cfg['IMAGE_SIZE']}
        train_dataset = get_segmentation_dataset(self.dataset_cfg['NAME'], root=self.dataset_cfg['ROOT'],split='train', mode='train', **data_kwargs)

        data_kwargs = {'transform': input_transform, 'base_size': self.eval_cfg['BASE_SIZE'], 'crop_size': self.eval_cfg['IMAGE_SIZE']}

        val_dataset = get_segmentation_dataset(self.dataset_cfg['NAME'], root=self.dataset_cfg['ROOT'], split='val', mode='val', **data_kwargs)
        self.iters_per_epoch = len(train_dataset) // (self.world_size * self.bs)
        self.max_iters = self.epochs * self.iters_per_epoch
        workers = 4

        self.train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=self.train_cfg['DDP'])
        train_batch_sampler = make_batch_data_sampler(self.train_sampler, self.bs, self.max_iters)
        val_sampler = make_data_sampler(val_dataset, False, self.train_cfg['DDP'])
        val_batch_sampler = make_batch_data_sampler(val_sampler, self.bs)

        train_loader = DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=workers,
                                            pin_memory=True)
        val_loader = DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=workers,
                                          pin_memory=True)

        return train_loader, val_loader


    def main(self):

        model = self.model

        # for epoch in range(self.epochs):
        time1 = time.time()
        model.train()
        # if self.train_cfg['DDP']: self.train_sampler.set_epoch(epoch)
        # pbar = tqdm(enumerate(self.trainloader), total=self.iters_per_epoch, desc=f"Epoch: [{epoch+1}/{self.epochs}] Iter: [{0}/{self.iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        train_loss = 0.0 
        best_mIoU = 0.0
        best_macc = 0.0
        for iterr, (img, lbl, _) in enumerate(self.train_loader):
            if iterr == 0 and self.gpu==0:
                print(lbl.min(), lbl.max())
            self.optimizer.zero_grad(set_to_none=True)
            # for optim in self.optimizer:
            #     optim.zero_grad(set_to_none=True)
                
            img = img.cuda(self.gpu, non_blocking=True)
            lbl = lbl.cuda(self.gpu, non_blocking=True)

            with autocast(enabled=self.train_cfg['AMP']):
                loss, logits = model(pixel_values=img, labels=lbl)
                # logits = torch.nn.functional.log_softmax(logits, dim=1)
                # loss = self.loss_fn(logits, lbl)

            self.scaler.scale(loss).backward()
            
            # for optim in self.optimizer:
            #     self.scaler.step(optim)
            self.scaler.step(self.optimizer)
            
            self.scaler.update()
            self.scheduler.step()
            # self.scheduler2.step()
            # torch.cuda.synchronize()
            # adjust_learning_rate(optimizers=self.optimizer, cur_iter=iterr, lr=self.lr, max_iter=self.max_iters)
            lr = self.scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            eta_seconds = ((time.time() - time1) / (iterr+1)) * (self.max_iters - iterr)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            # print(iterr+1, self.iters_per_epoch)

            if self.gpu == 0 and (iterr + 1) % (self.iters_per_epoch//2) ==0:
                self.logger.log(
                "Epoch: {:d}/{:d} | Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.7f} || Cost Time: {} || Estimated Time: {}".format(iterr//self.iters_per_epoch + 1, self.epochs,
                    iterr, self.max_iters, self.optimizer.param_groups[0]['lr'], train_loss / (iterr+1),
                    str(datetime.timedelta(seconds=int(time.time() - time1))), eta_string))

            if self.gpu == 0 and (iterr+1) % (self.iters_per_epoch*5) == 0:
                # print("yup")
                eval__stats = evaluate(model, self.val_loader, self.gpu, self.dataset_cfg['N_CLS'])
                miou = eval__stats[-1]
                macc = eval__stats[1]

                # self.writer.add_scalar('val/mIoU', miou, iterr//self.iters_per_epoch)
                self.logger.log(f"Epoch: [{iterr//self.iters_per_epoch+1}] \t Val miou: {miou}")
                model.train()

                if miou > best_mIoU:
                    best_mIoU = miou
                    torch.save(model.module.state_dict() if self.train_cfg['DDP'] else model.state_dict(), self.save_path + "/best_model_ckpt.pth")
                if macc > best_macc:
                    best_macc = macc
                print(f"Current mIoU: {miou} Best mIoU: {best_mIoU}")
                print(f"Current mAcc: {macc} Best mIoU: {best_macc}")

            if self.gpu==0 and (iterr + 1) % self.iters_per_epoch == 0:
                train_loss /= iterr+1
                # self.writer.add_scalar('train/loss', train_loss, (iterr + 1)//self.iters_per_epoch)
                train_loss = 0.0  # per epoch loss is Zero

        # self.writer.close()
        end = time.gmtime(time.time() - time1)

        table = [
            ['Best mIoU', f"{best_mIoU:.2f}"],
            ['Best mAcc', f"{best_macc:.2f}"],

            ['Total Training Time', time.strftime("%H:%M:%S", end)]
        ]
        if self.gpu == 0:
            print(tabulate(table, numalign='right'))

            self.logger.log(str(tabulate(table, numalign='right')))


    @classmethod
    def launch_from_args(cls, world_size, cfg):
        distributed=True
        if distributed:
            torch.multiprocessing.spawn(cls._exec_wrapper, args=(cfg,), nprocs=world_size, join=True)
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        cls.exec(*args, **kwargs)

    @classmethod
    def exec(cls, gpu, cfg):
        trainer = cls(gpu=gpu, cfg=cfg)
        trainer.main()
        trainer.cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml', help='Configuration file to use')
    parser.add_argument('--world_size', type=int, default=6, help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    Trainer.launch_from_args(args.world_size, cfg)

    # main(cfg, gpu, save_dir)
