# -*- coding: utf-8 -*-
# @Description:
#   Empty worker for new workers writing. This is an example.

# - Package Imports - #
from worker.worker import Worker


# - Coding Part - #
class ExpSampleWorker(Worker):
    def __init__(self, args):
        """
            Please add all parameters will be used in the init function.
        """
        super().__init__(args)

    def init_dataset(self):
        """
            Requires:
                self.train_dataset
                self.test_dataset
                self.res_writers  (self.create_res_writers)
        """
        raise NotImplementedError(':init_dataset() is not implemented by base class.')

    def init_networks(self):
        """
            Requires:
                self.networks (dict.)
            Keys will be used for network saving.
        """
        raise NotImplementedError(':init_networks() is not implemented by base class.')

    def init_losses(self):
        """
            Requires:
                self.loss_funcs (dict.)
            Keys will be used for avg_meter.
        """
        raise NotImplementedError(':init_losses() is not implemented by base class.')

    def data_process(self, idx, data):
        """
            Process data and move data to self.device.
            output will be passed to :net_forward().
        """
        raise NotImplementedError(':data_process() is not implemented by base class.')

    def net_forward(self, data):
        """
            How networks process input data and give out network output.
            The output will be passed to :loss_forward().
        """
        raise NotImplementedError(':net_forward() is not implemented by base class.')
    
    def loss_forward(self, net_out, data):
        """
            How loss functions process the output from network and input data.
            The output will be used with err.backward().
        """
        raise NotImplementedError(':loss_forward() is not implemented by base class.')

    def callback_after_train(self, epoch):
        raise NotImplementedError(':callback_after_train() is not implemented by base class.')

    def callback_save_res(self, data, net_out, dataset, res_writer):
        """
            The callback function for data saving.
            The data should be saved with the input.
            Please create new folders and save result.
        """
        raise NotImplementedError(':callback_save_res() is not implemented by base class.')

    def check_img_visual(self, **kwargs):
        """
            The img_visual callback entries.
            Modified this function can control the report strategy during training and testing.
            (Additional)
        """
        return super().check_img_visual(**kwargs)

    def callback_img_visual(self, data, net_out, tag, step):
        """
            The callback function for visualization.
            Notice that the loss will be automatically report and record to writer.
            For image visualization, some custom code is needed.
            Please write image to loss_writer and call flush().
        """
        raise NotImplementedError(':callback_img_visual() is not implemented by base class.')
