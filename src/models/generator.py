import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.resnet import BasicBlock
from math import sqrt
class ResBlk1D(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv1d(dim_in, dim_in, 5, 1, 2)
        self.conv2 = nn.Conv1d(dim_in, dim_out, 5, 1, 2)
        if self.normalize:
            self.norm1 = nn.BatchNorm1d(dim_in)
            self.norm2 = nn.BatchNorm1d(dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool1d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool1d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 5, 1, 2)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 5, 1, 2)
        if self.normalize:
            self.norm1 = nn.BatchNorm2d(dim_in)
            self.norm2 = nn.BatchNorm2d(dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class GenResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 5, 1, 2)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 5, 1, 2)
        self.norm1 = nn.BatchNorm2d(dim_in)
        self.norm2 = nn.BatchNorm2d(dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x):
        x = self.norm1(x)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        out = self._residual(x)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Avgpool(nn.Module):
    def forward(self, input):
        #input:B,C,H,W
        return input.mean([2, 3])

class AVAttention(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.softmax = nn.Softmax(2)
        self.k = nn.Linear(512, out_dim)
        self.v = nn.Linear(512, out_dim)
        self.q = nn.Linear(2560, out_dim)
        self.out_dim = out_dim
        dim = 20 * 64
        self.mel = nn.Linear(out_dim, dim)

    def forward(self, ph, g, len):
        #ph: B,S,512
        #g: B,C,F,T
        B, C, F, T = g.size()
        k = self.k(ph).transpose(1, 2).contiguous()   # B,256,S
        q = self.q(g.view(B, C * F, T).transpose(1, 2).contiguous())  # B,T,256

        att = torch.bmm(q, k) / math.sqrt(self.out_dim)    # B,T,S
        for i in range(att.size(0)):
            att[i, :, len[i]:] = float('-inf')
        att = self.softmax(att)  # B,T,S

        v = self.v(ph)  # B,S,256
        value = torch.bmm(att, v)  # B,T,256
        out = self.mel(value)  # B, T, 20*64
        out = out.view(B, T, F, -1).permute(0, 3, 2, 1)

        return out  #B,C,F,T
#mel-spectrogram -> linear spectrogram
class Postnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.postnet = nn.Sequential(
            nn.Conv1d(80, 128, 7, 1, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            ResBlk1D(128, 256),
            ResBlk1D(256, 256),
            ResBlk1D(256, 256),
            nn.Conv1d(256, 321, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        # x: B,1,80,T 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        x = x.squeeze(1)    # B, 80, t
        x = self.postnet(x)     # B, 321, T
        x = x.unsqueeze(1)  # B, 1, 321, T
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.ModuleList()
        self.g1 = nn.ModuleList()
        self.g2 = nn.ModuleList()
        self.g3 = nn.ModuleList()

        self.att1 = AVAttention(256)
        self.attconv1 = nn.Conv2d(128 + 64, 128, 5, 1, 2)
        self.att2 = AVAttention(256)
        self.attconv2 = nn.Conv2d(64 + 32, 64, 5, 1, 2)

        self.to_mel1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 1, 1, 0),
            nn.Tanh()
        )
        self.to_mel2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 1, 1, 0),
            nn.Tanh()
        )
        self.to_mel3 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.Tanh()
        )

        # bottleneck blocks
        self.decode.append(GenResBlk(512 + 128, 512))    # 20,T
        self.decode.append(GenResBlk(512, 256))
        self.decode.append(GenResBlk(256, 256))

        # up-sampling blocks
        self.g1.append(GenResBlk(256, 128))     # 20,T
        self.g1.append(GenResBlk(128, 128))
        self.g1.append(GenResBlk(128, 128))

        self.g2.append(GenResBlk(128, 64, upsample=True))  # 40,2T
        self.g2.append(GenResBlk(64, 64))
        self.g2.append(GenResBlk(64, 64))

        self.g3.append(GenResBlk(64, 32, upsample=True))  # 80,4T
        self.g3.append(GenResBlk(32, 32))
        self.g3.append(GenResBlk(32, 32))

    def forward(self, s, x, len):
        # s: B,512,T x: B,T,512
        s = s.transpose(1, 2).contiguous()
        n = torch.randn([x.size(0), 128, 20, x.size(1)]).cuda()  # B,128,20,T
        x = x.transpose(1, 2).contiguous().unsqueeze(2).repeat(1, 1, 20, 1)  # B, 512, 20, T
        x = torch.cat([x, n], 1)
        for block in self.decode:
            x = block(x)
        for block in self.g1:
            x = block(x)
        g1 = x.clone()
        c1 = self.att1(s, g1, len)
        x = self.attconv1(torch.cat([x, c1], 1))
        for block in self.g2:
            x = block(x)
        g2 = x.clone()
        c2 = self.att2(s, g2, len)
        x = self.attconv2(torch.cat([x, c2], 1))
        for block in self.g3:
            x = block(x)
        return self.to_mel1(g1), self.to_mel2(g2), self.to_mel3(x)

class Discriminator(nn.Module):
    def __init__(self, num_class=1, max_conv_dim=512, phase='1'):
        super().__init__()
        dim_in = 32
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 5, 1, 2)]

        if phase == '1':
            repeat_num = 2
        elif phase == '2':
            repeat_num = 3
        else:
            repeat_num = 4

        for _ in range(repeat_num): # 80,4T --> 40,2T --> 20,T --> 10,T/2 --> 5,T/4
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        self.main = nn.Sequential(*blocks)

        uncond = []
        uncond += [nn.LeakyReLU(0.2)]
        uncond += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        uncond += [nn.LeakyReLU(0.2)]
        uncond += [Avgpool()]
        uncond += [nn.Linear(dim_out, num_class)]
        self.uncond = nn.Sequential(*uncond)

        cond = []
        cond += [nn.LeakyReLU(0.2)]
        cond += [nn.Conv2d(dim_out + 512, dim_out, 5, 1, 2)]
        cond += [nn.LeakyReLU(0.2)]
        cond += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        cond += [nn.LeakyReLU(0.2)]
        cond += [Avgpool()]
        cond += [nn.Linear(dim_out, num_class)]
        self.cond = nn.Sequential(*cond)

    def forward(self, x, c, vid_max_length):
        # c: B,C,T
        f_len = final_length(vid_max_length)
        c = c.mean(2) #B,C
        c = c.unsqueeze(2).unsqueeze(2).repeat(1, 1, 5, f_len)
        out = self.main(x).clone()
        uout = self.uncond(out)
        out = torch.cat([out, c], dim=1)
        cout = self.cond(out)
        uout = uout.view(uout.size(0), -1)  # (batch, num_domains)
        cout = cout.view(cout.size(0), -1)  # (batch, num_domains)
        return uout, cout

class sync_Discriminator(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()

        self.frontend = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.PReLU(256)
        )

        self.Res_block = nn.Sequential(
            BasicBlock(256, 256)
        )

        self.Linear = nn.Linear(256 * 20, 512)
        self.temp = temp

    def forward(self, v_feat, aud, gen=False):
        # v_feat: B, S, 512
        a_feat = self.frontend(aud)
        a_feat = self.Res_block(a_feat)
        b, c, f, t = a_feat.size()
        a_feat = a_feat.view(b, c * f, t).transpose(1, 2).contiguous()  # B, T/4, 256 * F/4
        a_feat = self.Linear(a_feat)    # B, S, 512

        if gen:
            sim = torch.abs(F.cosine_similarity(v_feat, a_feat, 2)).mean(1)    #B, S
            loss = 5 * torch.ones_like(sim) - sim
        else:
            v_feat_norm = F.normalize(v_feat, dim=2)    #B,S,512
            a_feat_norm = F.normalize(a_feat, dim=2)    #B,S,512

            sim = torch.bmm(v_feat_norm, a_feat_norm.transpose(1, 2)) / self.temp #B,v_S,a_S

            nce_va = torch.mean(torch.diagonal(F.log_softmax(sim, dim=2), dim1=-2, dim2=-1), dim=1)
            nce_av = torch.mean(torch.diagonal(F.log_softmax(sim, dim=1), dim1=-2, dim2=-1), dim=1)

            loss = -1/2 * (nce_va + nce_av)

        return loss
def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self
def silu(x):
    return x * torch.sigmoid(x)
Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d
class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x
class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table
class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, n_cond_global=None):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        if n_cond_global is not None:
            self.conditioner_projection_global = Conv1d(n_cond_global, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step, conditioner_global=None):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        if conditioner_global is not None:
            y = y + self.conditioner_projection_global(conditioner_global)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip
class PriorGrad(nn.Module):
    def __init__(self):
        super().__init__()
        params = AttrDict(
            # Training params
            batch_size=16,
            learning_rate=2e-4,
            max_grad_norm=None,
            use_l2loss=True,

            # Data params
            sample_rate=22050,
            n_mels=80,
            n_fft=1024,
            hop_samples=256,
            fmin=0,
            fmax=8000,
            crop_mel_frames=62,  # PriorGrad keeps the previous open-source implementation

            # new data params for PriorGrad-vocoder
            use_prior=True,
            # optional parameters to additionally use the frame-level energy as the conditional input
            # one can choose one of the two options as below. note that only one can be set to True.
            condition_prior=False,  # whether to use energy prior as concatenated feature with mel. default is false
            condition_prior_global=False,
            # whether to use energy prior as global condition with projection. default is false
            # minimum std that clips the prior std below std_min. ensures numerically stable training.
            std_min=0.1,
            # whether to clip max energy to certain value. Affects normalization of the energy.
            # Lower value -> more data points assign to ~1 variance. so pushes latent space to higher variance regime
            # if None, no override, uses computed stat
            # for volume-normalized waveform with HiFi-GAN STFT, max energy of 4 gives reasonable range that clips outliers
            max_energy_override=4.,

            # Model params
            residual_layers=30,
            residual_channels=64,
            dilation_cycle_length=10,
            noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),  # [beta_start, beta_end, num_diffusion_step]
            inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],  # T>=50
            # inference_noise_schedule=[0.001, 0.01, 0.05, 0.2] # designed for for T=20
        )
        self.params = params
        self.use_prior = params.use_prior
        self.condition_prior = params.condition_prior
        self.condition_prior_global = params.condition_prior_global
        assert not (self.condition_prior and self.condition_prior_global),\
          "use only one option for conditioning on the prior"
        print("use_prior: {}".format(self.use_prior))
        self.n_mels = params.n_mels
        self.n_cond = None
        print("condition_prior: {}".format(self.condition_prior))
        if self.condition_prior:
            self.n_mels = self.n_mels + 1
            print("self.n_mels increased to {}".format(self.n_mels))
        print("condition_prior_global: {}".format(self.condition_prior_global))
        if self.condition_prior_global:
            self.n_cond = 1

        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        self.spectrogram_upsampler = SpectrogramUpsampler(self.n_mels)
        if self.condition_prior_global:
            self.global_condition_upsampler = SpectrogramUpsampler(self.n_cond)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(self.n_mels, params.residual_channels, 2 ** (i % params.dilation_cycle_length),
                          n_cond_global=self.n_cond)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

        print('num param: {}'.format(sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, audio, spectrogram, diffusion_step, global_cond=None):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        spectrogram = self.spectrogram_upsampler(spectrogram)
        if global_cond is not None:
            global_cond = self.global_condition_upsampler(global_cond)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, spectrogram, diffusion_step, global_cond)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x

def gan_loss(inputs, label=None):
    # non-saturating loss with R1 regularization
    l = -1 if label else 1
    return F.softplus(l*inputs).mean()

def final_length(vid_length):
    half = (vid_length // 2)
    quad = (half // 2)
    return quad
