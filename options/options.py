from .base_options import BaseOptions


class AdvanceOptions(BaseOptions):
    def initialize(self, parser, flag):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=400, help='# of iter at starting learning rate')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for optimizer')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor for SGD')
        parser.add_argument('--weight_decay', type=float, default=0.0005, help='momentum factor for optimizer')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=5000000, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lr_decay_epochs', type=int, default=100, help='multiply by a gamma every lr_decay_epoch epochs')
        parser.add_argument('--lr_gamma', type=float, default=0.9, help='gamma factor for lr_scheduler')
        parser.add_argument('--nP', type=int, default=500, help='number of points')
        parser.add_argument('--train_path', required=True, help='path to the training dataset')
        parser.add_argument('--test_path', required=True, help='path to the testing dataset')
        self.isTrain = flag
        return parser
