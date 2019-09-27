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
        parser.set_defaults(init_type='xavier')
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

        # self.latent_dim = opt.latent_dim
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 3*256*256, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids,
                                          vaeLike=True)
        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_E)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    # def get_z_random(self, batch_size, nz, random_type='gauss'):
    #     if random_type == 'uni':
    #         z = torch.rand(batch_size, nz) * 2.0 - 1.0
    #     elif random_type == 'gauss':
    #         z = torch.randn(batch_size, nz)
    #     return z.to(self.device)
    #
    # def encode(self, input_image):
    #     mu, logvar = self.netE.forward(input_image)
    #     std = logvar.mul(0.5).exp_()
    #     eps = self.get_z_random(std.size(0), std.size(1))
    #     z = eps.mul(std).add_(mu)
    #     return z, mu, logvar

    def forward(self):
        # z = torch.randn(self.real_A.size(0), self.latent_dim)  # torch.FloatTensor [1, 10]
        # z = torch.zeros(self.real_A.size(0), self.latent_dim)
        # z = self.real_B.view(self.real_B.size(0), -1)  #torch.cuda.FloatTensor [1, 3, 256, 256]
        z = self.netE.forward(self.real_B)
        # print(z.size())
        self.fake_B = self.netG(self.real_A, z)  # G_A(A)
        # self.fake_B = self.netG(self.real_A)

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
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_G = self.loss_G_GAN # + self.loss_G_L1 * self.opt.lambda_L1
        self.loss_G.backward()

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            # pred_fake = netD(fake)
            # loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
            loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), 0.9)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_B, self.netD, 1.0)#self.opt.lambda_GAN)
        # if self.opt.use_same_D:
        # self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.opt.lambda_GAN2)
        # else:
        #     self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.opt.lambda_GAN2)
        # # 2. KL loss
        # if self.opt.lambda_kl > 0.0:
        #     self.loss_kl = torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * (-0.5 * self.opt.lambda_kl)
        # else:
        #     self.loss_kl = 0
        # # 3, reconstruction |fake_B-real_B|
        # if self.opt.lambda_L1 > 0.0:
        #     self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        # else:
        #     self.loss_G_L1 = 0.0
        self.loss_G = self.loss_G_GAN # + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self):
        self.forward()      # compute fake images and reconstruction images.
        self.set_requires_grad(self.netD, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_EG()
        # self.backward_E()
        # self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        self.optimizer_E.step()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
