# latent model: the noise is an fixed image
import torch
import torch.nn as nn
import torchvision.models as models
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
# from . import networks_skip
from . import networks_resadd
from . import draw_featuremap


class LatentPretrainedSkipModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)
        parser.set_defaults(netG='LatentFilter')
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

        self.netG = networks_resadd.define_G(opt.input_nc, opt.output_nc, opt.ngf, 3*256*256, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD = networks_resadd.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.pretrainedmodel = models.vgg19(pretrained=True).to(self.device)
        self.feature = torch.nn.Sequential(*list(self.pretrainedmodel.children())[:1])
        self.pretrainedmodel.eval() # self.feature.eval()


        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks_resadd.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):
        z = self.feature(self.real_B) # [1,512,7,7]
        z_linear = z.view(z.size(0), -1)  # [1, 512*7*7]
        z_ = z_linear.clone()
        mean = torch.mean(z_[z_!=0]) # 求非0均值
        z_[z_ < mean] = 0
        self.fake_B = self.netG(self.real_A, z_)  # G_A(A)


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, 0.9)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):

        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D = self.backward_D_basic(self.netD, self.real_B, fake_B)

    def backward_G(self):

        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), 0.9)
        self.loss_G = self.loss_G_GAN
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
