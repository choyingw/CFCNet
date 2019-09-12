import os

class Visualizer():
    def __init__(self, opt):
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        
    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_depth_evaluation(self, message):
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
