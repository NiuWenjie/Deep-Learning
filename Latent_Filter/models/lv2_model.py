# latent model: the noise is an fixed image
import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Lv2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.set_defaults(direction='BtoA')
        parser.set_defaults(netG='LatentFilter')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['D', 'G']
        visual_names = ['real_A', 'real_B', 'fake_B']
        self.visual_names = visual_names

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.latent_dim = opt.latent_dim
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 3*256*256, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def get_z_random(self, batchSize, nz):
        z = torch.cuda.FloatTensor(batchSize, nz)
        z.copy_(torch.rand(batchSize, nz) * 2.0 - 1.0)
        return z

    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):
        # z = torch.randn(self.real_A.size(0), self.latent_dim)  # torch.FloatTensor [1, 10]
        # z = torch.zeros(self.real_A.size(0), self.latent_dim)
        # torch.cuda.FloatTensor [1, 3, 256, 256]
        self.z = self.real_B.view(self.real_B.size(0), -1)
        self.z2 = self.get_z_random(self.real_B.size(0), 3*256*256)
        # self.fake_B = self.netG(self.real_A, z)  # G_A(A)
        self.fake_B = self.netG(self.real_A, self.z)
        self.fake_B2 = self.netG(self.real_A, self.z2)
        # self.fake_B = self.netG(self.real_A)

    def backward_D_basic(self, netD, real, fake1 ,fake2):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, 0.9)
        # Fake
        # pred_fake1 = netD(fake1.detach())
        pred_fake1 = netD(fake1)
        pred_fake2 = netD(fake2.detach())
        loss_D_fake1 = self.criterionGAN(pred_fake1, False)
        loss_D_fake2 = self.criterionGAN(pred_fake2, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + (loss_D_fake1 + loss_D_fake2) * 0.5) * 0.5 # + (loss_D_real_L2 + loss_D_fake_L2) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):

        fake_B1 = self.fake_B_pool.query(self.fake_B)
        fake_B2 = self.fake_A_pool.query(self.fake_B2)
        self.loss_D = self.backward_D_basic(self.netD, self.real_B, fake_B1,fake_B2)

    def backward_G(self):

        self.loss_G_GAN1 = self.criterionGAN(self.netD(self.fake_B), 0.9)
        self.loss_G_GAN2 = self.criterionGAN(self.netD(self.fake_B2), 0.9)
        # mode seeking loss
        lz = torch.mean(torch.abs(self.fake_B2 - self.fake_B)) / torch.mean(torch.abs(self.z - self.z2))
        eps = 1 * 1e-5
        self.loss_lz = 1 / (lz + eps)
        self.loss_G = (self.loss_G_GAN1 + self.loss_G_GAN2) * 0.5 + self.loss_lz
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()      # compute fake images and reconstruction images.
        self.set_requires_grad(self.netD, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
