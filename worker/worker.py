# -*- coding: utf-8 -*-
# @Description:
#   The base class for learning process.

# - Package Imports - #
import configparser
from datetime import datetime
import torch
import numpy as np
import os
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import utils.pointerlib as plb


# - Coding Part - #
class Worker:
    def __init__(self, args):
        self.args = args

        # Set random seeds
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.time_keeper = plb.TimeKeeper()

        self.train_dir = None
        self.test_dir = None
        self.res_dir = None

        self.loss_writer = None
        self.log_file = None

        self.train_dataset = None  # For training
        self.test_dataset = None  # For testing

        self.networks = OrderedDict()
        self.network_static_list = list()
        self.optimizer = None
        self.scheduler = None
        self.loss_funcs = OrderedDict()
        self.avg_meters = OrderedDict()

        self.N = args.batch_num
        self.device = args.device
        self.train_shuffle = True
        self.n_iter = None
        self.history_best = [None, True]

        self.stopwatch = None
        self.status = "Unknown"

    def logging(self, outputs, tag='paras', step=None, log=True, save=False):
        # Only first gpu have logs
        if self.args.local_rank != 0:
            return

        # Default parameters
        step = self.args.epoch_start if step is None else step
        if not isinstance(outputs, list):
            outputs = [outputs]

        for output_str in outputs:
            print(output_str)
            if log and self.loss_writer is not None:
                self.loss_writer.add_text(f"{tag}", output_str, step)

        # Save to .log file
        if save and self.log_file is not None:
            with open(str(self.log_file), 'a') as file:
                log_title = f'{tag}-{step}:\n'
                file.write(log_title)
                for output_str in outputs:
                    file.write(output_str + '\n')

    def init_all(self):
        self.init_logger()
        self.init_dataset()
        self.init_networks()
        self.init_losses()
        self.init_optimizers()

    def init_logger(self):
        def str2path(input_str):
            return None if input_str == '' else Path(input_str)

        self.train_dir = str2path(self.args.train_dir)
        self.test_dir = str2path(self.args.test_dir)
        self.res_dir = Path(self.args.out_dir) / self.args.res_dir_name

        # Init summaryWriters under res_dir
        if self.args.local_rank == 0:  # Only save for 1st GPU
            log_path = self.res_dir / 'log'
            log_path.mkdir(parents=True, exist_ok=True)
            self.loss_writer = SummaryWriter(str(log_path), 'loss')
            self.log_file = log_path / 'my_log.log'
            self.logging(f'Writer path: {log_path}')

    def init_dataset(self):
        raise NotImplementedError(':init_dataset() is not implemented by base class.')

    def init_networks(self):
        raise NotImplementedError(':init_networks() is not implemented by base class.')

    def _net_parallel(self):
        for name in self.networks:
            self.networks[name] = self.networks[name].to(self.args.device)
            self.networks[name] = torch.nn.parallel.DistributedDataParallel(
                self.networks[name], device_ids=[self.args.local_rank], find_unused_parameters=False
            )

    def _net_load(self, epoch_num):
        if self.args.model_dir == '':
            return
        model_dir = Path(self.args.model_dir)
        for name in self.networks:
            if name in self.network_static_list:
                model_name = model_dir / f'{name}_static.pt'
            else:
                model_name = model_dir / f'{name}_e{epoch_num}.pt'

            if model_name.exists():
                state_dict = torch.load(model_name, map_location=f'cuda:{self.args.local_rank}')
                self.networks[name].load_state_dict(state_dict, strict=True)
                if name in self.network_static_list:
                    self.networks[name].eval()
                else:
                    self.networks[name].train()
                self.logging(f'Models loaded: {model_name}')

    def _net_save(self, epoch_num):
        # Only save for first GPU.
        if self.args.local_rank != 0:
            return

        save_model_dir = self.res_dir / 'model'
        save_model_dir.mkdir(parents=True, exist_ok=True)
        for name in self.networks:
            if name in self.network_static_list:
                continue

            model_name = save_model_dir / f'{name}_e{epoch_num}.pt'
            if self.args.data_parallel:
                state_dict = self.networks[name].module.state_dict()
            else:
                state_dict = self.networks[name].state_dict()
            torch.save(state_dict, model_name)
            self.logging(f'Model saved: {model_name}', tag='save', step=epoch_num)

            # Save best model with test
            if self.history_best[1]:
                model_best_name = save_model_dir / f'{name}_e{self.args.epoch_start}_best.pt'
                if self.args.data_parallel:
                    state_dict = self.networks[name].module.state_dict()
                else:
                    state_dict = self.networks[name].state_dict()
                torch.save(state_dict, model_best_name)
        self.history_best[1] = False

        if self.args.remove_history:
            if epoch_num > self.args.epoch_start:
                self._net_remove(epoch_num - 1)

    def _net_remove(self, epoch_num):
        save_model_dir = self.res_dir / 'model'
        for name in self.networks:
            if name in self.network_static_list:
                continue
            model_name = save_model_dir / f'{name}_e{epoch_num}.pt'
            if model_name.exists():
                os.remove(str(model_name))

    def init_losses(self):
        raise NotImplementedError(':init_losses() is not implemented by base class.')

    def init_optimizers(self):

        # Optimizer:
        adam_list = []
        for network in self.networks.values():
            adam_list.append({'params': network.parameters(), 'lr': self.args.lr})
        self.optimizer = torch.optim.Adam(adam_list)

        # Scheduler:
        if self.args.lr_step == 0:
            gamma = 1.0
            self.args.lr_step = 1
        elif self.args.lr_step > 0:
            gamma = 0.5
        else:  # For finding the proper learning rate.
            gamma = 10.0
            self.args.lr_step = - self.args.lr_step
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.args.lr_step, gamma=gamma, last_epoch=-1)
        self.logging(f'Learning rate: {self.args.lr} * {gamma} for every {self.args.lr_step}')

    def do(self):
        # Set initial things
        epoch_start = self.args.epoch_start
        if self.args.exp_type == 'train':
            epoch_end = epoch_start + 2 if self.args.debug_mode else self.args.epoch_end
            self.n_iter = int(epoch_start * len(self.train_dataset) / self.N)
        elif self.args.exp_type in ['eval', 'online']:
            epoch_end = epoch_start + 1
            self.n_iter = 0
        else:
            raise NotImplementedError(f'Wrong exp_type: {self.args.exp_type}')

        # Load network
        self._net_load(epoch_start - 1)
        if self.args.data_parallel:
            self._net_parallel()

        # Set average meters for logging
        self.avg_meters['Total'] = plb.EpochMeter('Total')
        for loss_name in self.loss_funcs:
            self.avg_meters[loss_name] = plb.EpochMeter(loss_name)
        self.logging(f'Loss meters: {self.avg_meters.keys()}')

        for epoch_num in range(epoch_start, epoch_end):
            if self.args.exp_type in ['train', 'online']:
                # lr = self.scheduler.get_last_lr()[0]
                # self.logging(f'Start training epoch {epoch_num}: lr={lr} <{self.time_keeper}>', tag='epoch', step=epoch_num)
                self.train_epoch(epoch=epoch_num)
                self.scheduler.step()
            self.callback_after_train(epoch_num)

            # self.logging(f'Start testing epoch {epoch_num}. <{self.time_keeper}>', tag='epoch', step=epoch_num)
            self.test_epoch(epoch=epoch_num)
            self.callback_after_test(epoch_num)

            # Save model
            if self.args.exp_type in ['train'] and epoch_num % self.args.model_stone == 0:
                self._net_save(epoch_num)

    def data_process(self, idx, data):
        raise NotImplementedError(':data_process() is not implemented by base class.')

    def net_forward(self, data):
        raise NotImplementedError(':net_forward() is not implemented by base class.')

    def loss_forward(self, net_out, data):
        raise NotImplementedError(':loss_forward() is not implemented by base class.')

    def loss_record(self, loss_name, return_val_only=True, **kwargs):
        """This function is used for average loss record."""
        ret = self.loss_funcs[loss_name](**kwargs)
        loss = ret
        if isinstance(ret, tuple):
            loss = ret[0]
        self.avg_meters[loss_name].update(loss, self.N)
        return loss if return_val_only else ret

    def check_save_res(self, epoch):
        """
            Only check if epoch_num reached the save_stone.
            Please inherit this function if you have other requirements.
        """
        if self.args.save_stone == 0:
            return False
        else:
            return (epoch + 1) % self.args.save_stone == 0

    def callback_save_res(self, epoch, data, net_out, dataset):
        raise NotImplementedError(':callback_save_res() is not implemented by base class.')

    def check_realtime_report(self, **kwargs):
        if self.loss_writer is None:
            return False

        if self.args.debug_mode:
            return True

        if self.status == 'Train':
            if self.n_iter % self.args.report_stone == 0:
                return True
        elif self.status == 'Eval':
            assert 'idx' in kwargs
            if kwargs['idx'] % self.args.report_stone == 0:
                return True
        else:
            raise NotImplementedError(f'Error status flag: {self.status}')

        return False

    def callback_realtime_report(self, batch_idx, batch_total, epoch, tag, step, log_flag=True):
        # Write loss
        total_loss = self.avg_meters['Total'].get_iter()
        for name in self.avg_meters.keys():
            if log_flag:
                self.loss_writer.add_scalar(f'{tag}-iter/{name}', self.avg_meters[name].get_iter(), self.n_iter)
            self.avg_meters[name].clear_iter()
        report_title = f'E{epoch:02d}B{batch_idx + 1:04d}/{batch_total:04d}-I{step:06d}'
        self.logging(f'{report_title}: loss = {total_loss:.4e}, [{self.time_keeper}]',
                     tag=self.args.exp_type, step=self.n_iter, log=log_flag)
        self.loss_writer.flush()

    def callback_epoch_report(self, epoch, tag, stopwatch):
        total_loss = self.avg_meters['Total'].get_epoch()
        for name in self.avg_meters.keys():
            self.loss_writer.add_scalar(f'{tag}-epoch/{name}', self.avg_meters[name].get_epoch(), epoch)
            self.avg_meters[name].clear_epoch()
        self.logging(f'[{self.status}]-Timings: {stopwatch}, total_loss={total_loss}', tag=tag, step=epoch)

        # Markdown best performance
        target_tag = 'eval' if not isinstance(self.test_dataset, list) else 'eval0'  # First
        if tag == target_tag:
            if self.history_best[0] is None or self.history_best[0] > total_loss:
                self.history_best = [total_loss, True]

        self.loss_writer.flush()

    def check_img_visual(self, **kwargs):
        if self.loss_writer is None:
            return False

        if self.args.debug_mode:
            return True

        if self.status == 'Train':
            if self.n_iter % self.args.img_stone == 0:
                return True
        elif self.status == 'Eval':
            assert 'idx' in kwargs
            if kwargs['idx'] % self.args.img_stone == 0:
                return True
        else:
            raise NotImplementedError(f'Error status flag: {self.status}')

        return False

    def callback_img_visual(self, data, net_out, tag, step):
        raise NotImplementedError(':callback_img_visual() is not implemented by base class.')

    def callback_after_train(self, epoch):
        # raise NotImplementedError(':callback_after_train() is not implemented by base class.')
        pass

    def callback_after_test(self, epoch):
        pass

    def train_epoch(self, epoch):
        self.stopwatch = plb.StopWatch()
        self.status = 'Train'

        if self.train_dataset is None:
            return

        #
        # Set dataloader and networks
        #
        train_sampler = None
        shuffle = self.train_shuffle
        if self.args.data_parallel:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            train_sampler.set_epoch(epoch)
            shuffle = None
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.N, shuffle=shuffle,
                                                   num_workers=self.args.num_workers, drop_last=True,
                                                   sampler=train_sampler)
        for name in self.networks:
            self.networks[name] = self.networks[name].to(self.device)
            self.networks[name].train()

        #
        # Main loop of epoch training
        #
        self.stopwatch.start('total')
        for idx, data in enumerate(train_loader):

            with self.stopwatch.record('data'):
                data = self.data_process(idx=idx, data=data)

            self.n_iter += 1
            self.optimizer.zero_grad()

            with self.stopwatch.record('forward'):
                output = self.net_forward(data=data)

            with self.stopwatch.record('loss'):
                err = self.loss_forward(net_out=output, data=data)

            with self.stopwatch.record('backward'):
                err.backward()

            with self.stopwatch.record('optimizer'):
                self.optimizer.step()

            #
            # Reporting part
            #
            if self.check_save_res(epoch):
                self.callback_save_res(epoch=epoch, data=data, net_out=output, dataset=self.train_dataset)

            # Report in real-time
            if self.check_realtime_report():
                self.callback_realtime_report(batch_idx=idx, batch_total=len(train_loader), epoch=epoch,
                                              tag='train', step=self.n_iter)

            # Draw image
            if self.check_img_visual():
                self.callback_img_visual(data=data, net_out=output, tag='train', step=self.n_iter)

            if self.args.debug_mode:
                break

        self.stopwatch.stop('total')

        # Print output
        if self.loss_writer is not None:
            self.callback_epoch_report(epoch, tag='train', stopwatch=self.stopwatch)

    def test_epoch(self, epoch):
        self.status = 'Eval'

        def apply_test(test_dataset, epoch, tag):
            if test_dataset is None:
                return

            test_sampler = None
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                      num_workers=self.args.num_workers, drop_last=True,
                                                      sampler=test_sampler)
            for name in self.networks:
                self.networks[name] = self.networks[name].to(self.device)
                self.networks[name].eval()

            with torch.no_grad():
                self.stopwatch = plb.StopWatch()
                self.stopwatch.start('total')
                for idx, data in enumerate(test_loader):

                    with self.stopwatch.record('data'):
                        data = self.data_process(idx=idx, data=data)

                    with self.stopwatch.record('forward'):
                        net_out = self.net_forward(data=data)

                    if self.check_save_res(epoch):
                        self.callback_save_res(epoch=epoch, data=data, net_out=net_out, dataset=test_dataset)

                    # Draw image
                    if self.check_img_visual(idx=idx):
                        self.callback_img_visual(data=data, net_out=net_out, tag=tag, step=epoch)

                    # Report in real-time
                    if self.check_realtime_report(idx=idx):
                        self.callback_realtime_report(batch_idx=idx, batch_total=len(test_loader), epoch=epoch,
                                                      tag='eval', step=self.n_iter, log_flag=False)

                    if self.args.debug_mode:
                        break

                self.stopwatch.stop('total')

                # Report
                if self.loss_writer is not None:
                    self.callback_epoch_report(epoch=epoch, tag=tag, stopwatch=self.stopwatch)

        if not isinstance(self.test_dataset, list):
            apply_test(self.test_dataset, epoch, 'eval')
        else:
            for i, test_dataset in enumerate(self.test_dataset):
                apply_test(test_dataset, epoch, f'eval{i}')
