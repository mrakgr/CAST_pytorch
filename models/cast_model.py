import itertools
import torch
from .base_model import BaseModel
from . import net
import util.util as util
from util.image_pool import ImagePool
import torch.nn as nn
from torch.nn import init

class CASTModel(BaseModel):
    """ This class implements CAST model.
    This code is inspired by DCLGAN and CycleGAN.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CAST """
        parser.add_argument('--CAST_mode', type=str, default="CAST", choices='CAST')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CAST.
        if opt.CAST_mode.lower() == "cast":
            pass
            # parser.set_defaults(nce_idt=False, lambda_NCE=2.0)
        else:
            raise ValueError(opt.CAST_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['AE','Dec_A']

        # define networks (both generator and discriminator)
        
        vgg = net.vgg
        vgg = nn.Sequential(*list(vgg.children())[:31])
        self.netAE = net.ADAIN_Encoder(vgg, self.gpu_ids)
        self.netDec_A = net.Decoder(self.gpu_ids)
        
    def forward(self,input,style_weight=1):
        AtoB = self.opt.direction == 'AtoB'
        real_A = input['A' if AtoB else 'B'].to(self.device)
        real_B = input['B' if AtoB else 'A'].to(self.device)
        return self.netDec_A(self.netAE(real_A, real_B, style_weight))
    
