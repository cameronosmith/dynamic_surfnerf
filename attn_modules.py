import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import numpy as np
import torch
import numpy as np
import util

from einops import rearrange, repeat, reduce
from torch import einsum

import conv_modules
import custom_layers
import geometry

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

from torch import einsum
class CrossAttn_(nn.Module):
    def __init__(self, ch, heads=8, dim_head=64, y_ch = None, no_kvs=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.ch = ch
        self.no_kvs = no_kvs
        if y_ch is None: y_ch = ch

        self.to_q = nn.Linear(y_ch, inner_dim, bias=False)
        if no_kvs is False:
            self.to_kv = nn.Linear(ch, inner_dim * 2, bias=False)
        self.proj = nn.Linear(inner_dim, y_ch)

        self.out = nn.Sequential(
            nn.Linear(y_ch, int(4*ch)),
            nn.GELU(),
            nn.Linear(int(4*ch), y_ch)
        )

        self.ln_1_y = nn.LayerNorm([y_ch])
        self.ln_1_x = nn.LayerNorm([2*ch if no_kvs else ch])
        self.ln_2 = nn.LayerNorm([y_ch])

    # x is the image patches and y is the cls tokens, for ex.
    def forward(self, x, y, attn_mask=None,return_heads=False, return_attn=False,softmax_axis=-1,max_agg=False,use_residual=True):

        x_ln = self.ln_1_x(x)
        y_ln = self.ln_1_y(y)

        h = self.heads

        q = self.to_q(y_ln)
        k, v = (x_ln if self.no_kvs else self.to_kv(x_ln)).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=softmax_axis)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        out = self.proj(out) + y
        out = self.out(self.ln_2(out)) + out

        return (out,rearrange(attn,"(b h) n d -> b n h d",h=h)) if return_attn else out

class LightFieldModel(nn.Module):
    def __init__(self, latent_dim, parameterization='plucker', network='relu',
                 fit_single=False, conditioning='hyper', depth=False, alpha=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_hidden_units_phi = 256
        self.fit_single = fit_single
        self.parameterization = parameterization
        self.conditioning = conditioning
        self.depth = depth
        self.alpha = alpha

        out_channels = 3

        if self.depth:
            out_channels += 1
        if self.alpha:
            out_channels += 1
            self.background = torch.ones((1, 1, 1, 3)).cuda()

        if self.fit_single or conditioning in ['hyper', 'low_rank']:
            if network == 'relu':
                self.phi = custom_layers.FCBlock(hidden_ch=self.num_hidden_units_phi, num_hidden_layers=6,
                                                 in_features=6, out_features=out_channels, outermost_linear=True, norm='layernorm_na')
            elif network == 'siren':
                omega_0 = 30.
                self.phi = custom_layers.Siren(in_features=6, hidden_features=256, hidden_layers=8,
                                               out_features=out_channels, outermost_linear=True, hidden_omega_0=omega_0,
                                               first_omega_0=omega_0)
        elif conditioning == 'concat':
            self.phi = nn.Sequential(
                nn.Linear(6+self.latent_dim, self.num_hidden_units_phi),
                custom_layers.ResnetBlockFC(size_in=self.num_hidden_units_phi, size_out=self.num_hidden_units_phi,
                                            size_h=self.num_hidden_units_phi),
                custom_layers.ResnetBlockFC(size_in=self.num_hidden_units_phi, size_out=self.num_hidden_units_phi,
                                            size_h=self.num_hidden_units_phi),
                custom_layers.ResnetBlockFC(size_in=self.num_hidden_units_phi, size_out=self.num_hidden_units_phi,
                                            size_h=self.num_hidden_units_phi),
                nn.Linear(self.num_hidden_units_phi, 3)
            )

        if not self.fit_single:
            if conditioning=='hyper':
                self.hyper_phi = hyperlayers.HyperNetwork(hyper_in_features=self.latent_dim,
                                                          hyper_hidden_layers=1,
                                                          hyper_hidden_features=self.latent_dim,
                                                          hypo_module=self.phi)
            elif conditioning=='low_rank':
                self.hyper_phi = hyperlayers.LowRankHyperNetwork(hyper_in_features=self.latent_dim,
                                                                 hyper_hidden_layers=1,
                                                                 hyper_hidden_features=512,
                                                                 hypo_module=self.phi,
                                                                 nonlinearity='leaky_relu')

        print(self.phi)
        print(np.sum(np.prod(param.shape) for param in self.phi.parameters()))

    def get_light_field_function(self, z=None):
        if self.fit_single:
            phi = self.phi
        elif self.conditioning in ['hyper', 'low_rank']:
            phi_weights = self.hyper_phi(z)
            phi = lambda x: self.phi(x, params=phi_weights)
        elif self.conditioning == 'concat':
            def phi(x):
                b, n_pix = x.shape[:2]
                z_rep = z.view(b, 1, self.latent_dim).repeat(1, n_pix, 1)
                return self.phi(torch.cat((z_rep, x), dim=-1))
        return phi

    def get_query_cam(self, input):
        query_dict = input['query']
        pose = query_dict["cam2world"].flatten(end_dim=1)
        intrinsics = query_dict["intrinsics"].flatten(end_dim=1)
        uv = query_dict["uv"].float().flatten(end_dim=1)
        return pose, intrinsics, uv

    def forward(self, input, val=False, compute_depth=False, timing=False):
        out_dict = {}
        query = input['query']
        b, n_ctxt = query["uv"].shape[:2]
        n_qry, n_pix = query["uv"].shape[1:3]

        if not self.fit_single:
            if 'z' in input:
                z = input['z']
            else:
                z = self.get_z(input)

            out_dict['z'] = z
            z = z.view(b * n_qry, self.latent_dim)

        query_pose, query_intrinsics, query_uv = self.get_query_cam(input)

        if self.parameterization == 'plucker':
            light_field_coords = geometry.plucker_embedding(query_pose, query_uv, query_intrinsics)
        else:
            ray_origin = query_pose[:, :3, 3][:, None, :]
            ray_dir = geometry.get_ray_directions(query_uv, query_pose, query_intrinsics)
            intsec_1, intsec_2 = geometry.ray_sphere_intersect(ray_origin, ray_dir, radius=100)
            intsec_1 = F.normalize(intsec_1, dim=-1)
            intsec_2 = F.normalize(intsec_2, dim=-1)

            light_field_coords = torch.cat((intsec_1, intsec_2), dim=-1)
            out_dict['intsec_1'] = intsec_1
            out_dict['intsec_2'] = intsec_2
            out_dict['ray_dir'] = ray_dir
            out_dict['ray_origin'] = ray_origin

        light_field_coords.requires_grad_(True)
        out_dict['coords'] = light_field_coords.view(b*n_qry, n_pix, 6)

        lf_function = self.get_light_field_function(None if self.fit_single else z)
        out_dict['lf_function'] = lf_function

        if timing: t0 = time.time()
        lf_out = lf_function(out_dict['coords'])
        if timing: t1 = time.time(); total_n = t1 - t0; print(f'{total_n}')

        rgb = lf_out[..., :3]

        if self.depth:
            depth = lf_out[..., 3:4]
            out_dict['depth'] = depth.view(b, n_qry, n_pix, 1)

        rgb = rgb.view(b, n_qry, n_pix, 3)

        if self.alpha:
            alpha = lf_out[..., -1:].view(b, n_qry, n_pix, 1)
            weight = 1 - torch.exp(-torch.abs(alpha))
            rgb = weight * rgb + (1 - weight) * self.background
            out_dict['alpha'] = weight

        if compute_depth:
            with torch.enable_grad():
                lf_function = self.get_light_field_function(z)
                depth = util.light_field_depth_map(light_field_coords, query_pose, lf_function)['depth']
                depth = depth.view(b, n_qry, n_pix, 1)
                out_dict['depth'] = depth

        out_dict['rgb'] = rgb
        return out_dict


class LFAutoDecoder(LightFieldModel):
    def __init__(self, latent_dim, num_instances, parameterization='plucker', **kwargs):
        super().__init__(latent_dim=latent_dim, parameterization=parameterization, **kwargs)
        self.num_instances = num_instances

        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

    def get_z(self, input, val=False):
        instance_idcs = input['query']["instance_idx"].long()
        z = self.latent_codes(instance_idcs)
        return z


class LFEncoder(LightFieldModel):
    def __init__(self, latent_dim, num_instances, parameterization='plucker', conditioning='hyper'):
        super().__init__(latent_dim, parameterization, conditioning='low_rank')
        self.num_instances = num_instances
        self.encoder = conv_modules.Resnet18(c_dim=latent_dim)

    def get_z(self, input, val=False):
        n_qry = input['query']['uv'].shape[1]
        rgb = util.lin2img(util.flatten_first_two(input['context']['rgb']))
        z = self.encoder(rgb)
        z = z.unsqueeze(1).repeat(1, n_qry, 1)
        z *= 1e-2
        return z


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class simple_CrossAttention(nn.Module):

    def __init__(self, slot_dim):
        super().__init__()

        self.kv_cross=nn.ModuleList([nn.Linear(slot_dim,slot_dim) for _ in range(2)]).cuda()
        self.q_cross=nn.Linear(slot_dim,slot_dim).cuda()
        self.kqv_self=nn.ModuleList([nn.Linear(slot_dim,slot_dim) for _ in range(3)]).cuda()
        self.proj=nn.Linear(slot_dim,slot_dim).cuda()

        self.out_cross = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim)
        )

        self.ln_1 = nn.LayerNorm([slot_dim])
        self.ln_2 = nn.LayerNorm([slot_dim])
        self.apply(init_weights)

        self.scale=slot_dim**(-0.5)

    # y is query coord information, x is img patches
    def forward(self, x, y):

        key,val=[f(x) for f in self.kv_cross]
        query=self.q_cross(y)

        A=(torch.einsum("btc,blc->btl",[key,query])*self.scale).softmax(1)
        R=torch.einsum("btc,btl->blc",[val,A])
        out=self.ln_1(self.proj(R))+y
        out=self.ln_2(self.out_cross(out)+out)
        return out


def unfold_non_overlapping(x, block_size):
    return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=block_size, p2=block_size)


class PatchedSelfAttention(nn.Module):
    def __init__(self, channels, neighborhood_size=4, num_heads=4, head_dim=64, shift_equivariant=True, down=False,
                 halo=None, coordinates=True, pos_encoding=True, coord_dim=0, out_ch=None, *args, **kwargs):
        super().__init__()
        self.ch = channels
        self.ks = neighborhood_size

        self.nh = num_heads
        self.hd = head_dim
        self.shift_equi = shift_equivariant
        self.down = down
        self.halo = halo
        self.coordinates = coordinates
        self.down_mode = 'token' # token, average, sub
        self.orig_coord_dim = coord_dim

        if self.coordinates:
            assert coord_dim > 0, "need to pass a coord_dim if coordinates are to be used"

        if pos_encoding:
            self.pe = custom_layers.PositionalEncoding(input_dim=coord_dim, include_input=True)
            self.coord_dim = self.pe.num_encoding_functions*2*coord_dim+coord_dim
        else:
            self.pe = lambda x: x
            self.coord_dim = coord_dim

        if self.down:
            self.down_token = nn.Parameter(torch.zeros((1, 1, 1, self.nh*self.hd)))
            nn.init.kaiming_normal_(self.down_token, a=0.0, nonlinearity='relu', mode='fan_in')

        if self.nh == 1:
            assert(head_dim==channels), "If not multi-head, number of channels needs to be equivalent to head dimension"

        in_ch = self.ch
        if self.coordinates:
            in_ch += self.coord_dim
        self.to_kv = nn.Linear(in_ch, self.hd*self.nh*2, bias=False)
        self.query_embed = nn.Linear(in_ch, self.hd * self.nh, bias=False)

        if self.nh > 1:
            self.multi_out = nn.Linear(self.hd*self.nh, self.ch)

        if out_ch is None:
            out_ch = self.ch

        self.out = nn.Sequential(
            nn.Linear(self.ch, int(4*self.ch)),
            nn.GELU(),
            nn.Linear(int(4*self.ch), out_ch)
        )

        self.ln_1 = nn.LayerNorm([self.ch])
        self.ln_2 = nn.LayerNorm([self.ch])

    def forward(self, inp, coords):
        b, _, h, w = inp.shape

        inp_unf_van = unfold_non_overlapping(inp, self.ks)
        inp_unf = self.ln_1(inp_unf_van)
        coords_unf = unfold_non_overlapping(coords, self.ks)

        if self.coordinates:
            if self.shift_equi:
                coords_unf -= coords_unf.mean(dim=-2, keepdims=True)  # center each of the patches

            coords_unf_pe = self.pe(coords_unf)
            inp_unf = torch.cat((coords_unf_pe, inp_unf), dim=-1)

        if self.halo is not None:
            kv_inp = F.unfold(inp, kernel_size = self.ks + self.halo * 2, stride = self.ks, padding=self.halo)
            kv_inp = rearrange(kv_inp, 'b (c k) i -> b i k c', c=self.ch)

            if self.coordinates:
                coords_kv_inp = F.unfold(coords, kernel_size = self.ks + self.halo * 2, stride = self.ks, padding=self.halo)
                coords_kv_inp = rearrange(coords_kv_inp, 'b (c k) i -> b i k c', c=self.orig_coord_dim)

                if self.shift_equi:
                    coords_kv_inp -= coords_kv_inp.mean(dim=-2, keepdims=True) # center each of the patches

                coords_kv_inp = self.pe(coords_kv_inp)
                kv_inp = torch.cat((kv_inp, coords_kv_inp), dim=-1)
        else:
            kv_inp = inp_unf

        k, v = self.to_kv(kv_inp).chunk(2, dim=-1) # (b, h*w/ks**2, ks*ks, self.nh*self.hd)

        if self.down: # subsample patches
            if self.down_mode == 'token':
                inp_unf_van = reduce(inp_unf_van, 'b n k d -> b n d', 'mean').unsqueeze(-2)
                coords = reduce(coords_unf, 'b n k d -> b n d', 'mean')
                q = repeat(self.down_token, 'b hw k ch -> (b b1) (hw hw1) k ch', b1=b, hw1=int(h*w/self.ks**2)) # (b, h*w/ks**2, 1, self.nh*self.hd)
            elif self.down_mode == 'average':
                q = self.query_embed(inp_unf_van) # (b, h*w/ks**2, ks**2, self.nh*self.hd)
                q = reduce(q, 'b n k d -> b n d', 'mean').unsqueeze(-2)
                inp_unf_van = reduce(inp_unf_van, 'b n k d -> b n d', 'mean').unsqueeze(-2)
                coords = reduce(coords_unf, 'b n k d -> b n d', 'mean')
            elif self.down_mode == 'sub':
                inp_unf_van = inp_unf_van[..., :1, :]
                coords = coords_unf[..., 0, :]
                q = self.query_embed(inp_unf) # (b, h*w/ks**2, ks**2, self.nh*self.hd)
        else:
            q = self.query_embed(inp_unf) # (b, h*w/ks**2, ks**2, self.nh*self.hd)

        q, k, v = map(lambda t: rearrange(t, 'b n k (h d) -> (b h) n k d', h=self.nh), (q, k, v))

        sim = einsum('b m j d, b m k d -> b m j k', q, k)
        attn = (sim / np.sqrt(self.hd)).softmax(dim=-1)

        out = einsum('b m j k, b m k d -> b m j d', attn, v)

        # concatenate heads
        out = rearrange(out, '(b h) m j d -> b m j (h d)', h=self.nh)
        out = self.multi_out(out)

        out = out + inp_unf_van[..., :self.ch]
        out = self.out(self.ln_2(out)) + out

        if self.down:
            out = rearrange(out.squeeze(dim=-2), 'b (h w) c -> b c h w', h=(h // self.ks), w=(w // self.ks))
            coords = rearrange(coords, 'b (h w) c -> b c h w', h=(h // self.ks), w=(w // self.ks))
        else:
            out = rearrange(out, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', b=b, h=(h // self.ks), w=(w // self.ks),
                            p1=self.ks, p2=self.ks)
        return out, coords


class GlobalAttention(nn.Module):
    def __init__(self, ch, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.ch = ch

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.proj = nn.Linear(inner_dim, ch)

        self.out = nn.Sequential(
            nn.Linear(ch, int(4*ch)),
            nn.GELU(),
            nn.Linear(int(4*ch), ch)
        )

        self.ln_1 = nn.LayerNorm([self.ch])
        self.ln_2 = nn.LayerNorm([self.ch])

    def forward(self, x, coords=None, context=None, ):
        b, _, height, width = x.shape

        x = rearrange(x, 'b ch h w -> b (h w) ch')
        x_ln = self.ln_1(x)

        h = self.heads

        q = self.to_q(x_ln)
        context = default(context, x_ln)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        out = self.proj(out) + x
        out = self.out(self.ln_2(out)) + out

        out = rearrange(out, 'b (h w) c -> b c h w', h=height, w=width)
        return out#, coords


class CrossAttention(nn.Module):
    def __init__(self, latent_dim, num_heads, head_dim, kv_in_dim, query_in_dim):
        super().__init__()

        self.ch = latent_dim
        self.nh = num_heads
        self.hd = head_dim

        self.to_kv = nn.Linear(kv_in_dim, self.hd * self.nh * 2, bias=False)
        self.q_emb = nn.Linear(query_in_dim, self.hd * self.nh, bias=False)

        if self.nh > 1:
            self.mh_proj = nn.Linear(self.hd * self.nh, self.ch)

        self.out = nn.Sequential(
            nn.Linear(self.ch, self.ch),
            nn.GELU(),
            nn.Linear(self.ch, self.ch)
        )

        self.ln_1 = nn.LayerNorm([self.ch])
        self.ln_2 = nn.LayerNorm([self.ch])
        self.apply(init_weights)

    def attention(self, k, v, q, query_in):
        sim = einsum('b n d, b n k d -> b n k', q, k)
        attn = (sim / np.sqrt(self.hd)).softmax(dim=-1)

        out = einsum('b n k, b n k d -> b n d', attn, v)

        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.nh)
        out = self.mh_proj(out)

        out = self.ln_1(out) + query_in
        out = self.ln_2(self.out(out) + out)
        return out


class PatchedCrossAttention(CrossAttention):
    def __init__(self, nh_size, **kwargs):
        super().__init__(**kwargs)
        self.ks = nh_size

    def forward(self, kv_in, query_in, query_coords):
        feat_h = kv_in.shape[-2]

        q = self.q_emb(query_in)
        q = rearrange(q, 'b n (nh hd) -> (b nh) n hd', nh=self.nh)

        if feat_h > self.ks:
            kv_in = F.unfold(kv_in, kernel_size=self.ks, stride=1, padding=self.ks // 2)  # (b, ch*ks**2, feat_h*feat_w)
            kv_in = rearrange(kv_in, 'b ch (feat_h feat_w) -> b ch feat_h feat_w', feat_h=feat_h)  # (b, ch*ks**2, feat_h, feat_w)
        else:
            kv_in = rearrange(kv_in, 'b ch feat_h feat_w -> b (ch feat_h feat_w)').unsqueeze(-1).unsqueeze(
                -1)  # (b, ch*feat_h*feat_w, 1, 1)

        neighborhoods = F.grid_sample(kv_in, query_coords.flip(-1).unsqueeze(1),
                                      mode='nearest', align_corners=False)[:, :, 0, :]  # (b, ch*ks**2, n_query)
        neighborhoods = rearrange(neighborhoods, 'b (c k) n_query -> b n_query k c', c=self.ch)

        k, v = self.to_kv(neighborhoods).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n k (h d) -> (b h) n k d', h=self.nh), (k, v))

        out = self.attention(k, v, q, query_in)
        return out


class GlobalCrossAttention(CrossAttention):
    def forward(self, kv_in, query_in, query_coords):
        q = self.q_emb(query_in)
        q = rearrange(q, 'b n (nh hd) -> (b nh) n hd', nh=self.nh)

        kv_in = rearrange(kv_in, 'b ch feat_h feat_w -> b (feat_h feat_w) ch').unsqueeze(1)

        k, v = self.to_kv(kv_in).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n k (h d) -> (b h) n k d', h=self.nh), (k, v))

        out = self.attention(k, v, q, query_in)
        return out


class MultiScaleDecoder(nn.Module):
    def __init__(self, channels, scale_idcs, num_heads=8, coord_dim=2, head_dim=64,
                 neighborhood_size=3, patched=True, concat_query_coords=False, pos_encoding=True):
        super().__init__()
        self.ch = channels

        self.nh = num_heads
        self.cd = coord_dim
        self.hd = head_dim
        self.si = scale_idcs
        self.ks = neighborhood_size
        self.patched = patched
        self.concat_query_coords = concat_query_coords

        if pos_encoding:
            self.pe = custom_layers.PositionalEncoding(input_dim=coord_dim, include_input=True)
            self.cd = self.pe.num_encoding_functions*2*coord_dim+coord_dim
        else:
            self.pe = lambda x: x
            self.cd = coord_dim

        if self.nh == 1:
            assert(head_dim==channels), "If not multi-head, number of channels needs to be equivalent to head dimension"

        if self.concat_query_coords:
            self.init_proj = nn.Sequential(
                nn.Linear(self.ch + self.cd, self.ch),
                nn.GELU(),
            )

        self.attention = nn.ModuleList()
        for i in range(len(self.si)):
            query_in_dim = self.ch

            if self.patched:
                self.attention += [
                    PatchedCrossAttention(latent_dim=self.ch, num_heads=self.nh, head_dim=self.hd,
                                          kv_in_dim=self.ch, query_in_dim=query_in_dim, nh_size=self.ks)
                ]
            else:
                self.attention += [
                    GlobalCrossAttention(latent_dim=self.ch, num_heads=self.nh, head_dim=self.hd,
                                         kv_in_dim=self.ch, query_in_dim=query_in_dim)
                ]

        self.apply(init_weights)

    def forward(self, multi_scale_inp, query_coords):
        b, n, ch = query_coords.shape
        scale_embeddings = []

        query = multi_scale_inp[-1].squeeze(-1).squeeze(-1).unsqueeze(1)
        query = repeat(query, 'b n ch -> b (n n1) ch', n1=n)  # (b, h*w/ks**2, 1, self.nh*self.hd)

        if self.concat_query_coords:
            query = torch.cat((query, self.pe(query_coords)), dim=-1)
            query = self.init_proj(query)

        scale_feats = [multi_scale_inp[idx] for idx in self.si]

        for feats, att in zip(scale_feats[::-1], self.attention):
            query = att(kv_in=feats, query_in=query, query_coords=query_coords)
            scale_embeddings.append(query)

        return scale_embeddings


class MultiscaleImgEncoder(nn.Module):
    def __init__(self, sidelength, in_channels=3, coord_dim=2, mid_ch=256):
        super().__init__()
        self.ic = in_channels
        self.cd = coord_dim
        self.mc = mid_ch

        self.in_embed = nn.Conv2d(in_channels, mid_ch, 1, padding=0)

        self.layers = nn.ModuleList()
        for i in range(int(np.log2(sidelength) - 5)): # after this loop, sidelength is 32
            self.layers.extend([
                PatchedSelfAttention(channels=mid_ch, halo=1, coordinates=i==0, coord_dim=2, num_heads=8, neighborhood_size=4,
                                     shift_equivariant=False, pos_encoding=True),
                PatchedSelfAttention(channels=mid_ch, halo=1, coordinates=False, coord_dim=2, num_heads=8, neighborhood_size=4),
                PatchedSelfAttention(channels=mid_ch, halo=1, coordinates=False, coord_dim=2, num_heads=8, neighborhood_size=4),
                PatchedSelfAttention(channels=mid_ch, halo=None, coordinates=False, coord_dim=2, num_heads=4, neighborhood_size=2, down=True),
            ])

        self.layers.extend([
            PatchedSelfAttention(channels=mid_ch, halo=1, coordinates=False, coord_dim=2, num_heads=8, neighborhood_size=4),
            PatchedSelfAttention(channels=mid_ch, halo=1, coordinates=False, coord_dim=2, num_heads=8, neighborhood_size=4),
            PatchedSelfAttention(channels=mid_ch, halo=1, coordinates=False, coord_dim=2, num_heads=8, neighborhood_size=4),
            PatchedSelfAttention(channels=mid_ch, halo=None, coordinates=False, coord_dim=2, num_heads=4, neighborhood_size=2, down=True),
            GlobalAttention(ch=mid_ch, query_dim=mid_ch, heads=8, dim_head=64),
            GlobalAttention(ch=mid_ch, query_dim=mid_ch, heads=8, dim_head=64),
            GlobalAttention(ch=mid_ch, query_dim=mid_ch, heads=8, dim_head=64),
            PatchedSelfAttention(channels=mid_ch, halo=None, coordinates=False, coord_dim=2, num_heads=8, neighborhood_size=4, down=True),
            GlobalAttention(ch=mid_ch, query_dim=mid_ch, heads=8, dim_head=64),
            GlobalAttention(ch=mid_ch, query_dim=mid_ch, heads=8, dim_head=64),
            GlobalAttention(ch=mid_ch, query_dim=mid_ch, heads=8, dim_head=64),
            PatchedSelfAttention(channels=mid_ch, halo=None, coordinates=False, coord_dim=2, num_heads=8, neighborhood_size=2, down=True),
            GlobalAttention(ch=mid_ch, query_dim=mid_ch, heads=8, dim_head=64),
            GlobalAttention(ch=mid_ch, query_dim=mid_ch, heads=8, dim_head=64),
            GlobalAttention(ch=mid_ch, query_dim=mid_ch, heads=8, dim_head=64),
            PatchedSelfAttention(channels=mid_ch, halo=None, coordinates=False, coord_dim=2, num_heads=8, neighborhood_size=2, down=True),
        ])

        self.apply(init_weights)

    def forward(self, input, coords):
        x = input
        c = coords

        b, n, ch = x.shape
        x = rearrange(x, 'b (h w) ch -> b ch h w', h=int(np.sqrt(n)))
        c = rearrange(c, 'b (h w) ch -> b ch h w', h=int(np.sqrt(n)))

        x = self.in_embed(x)

        multi_scale_feats = []
        multi_scale_coords = []
        for i, layer in enumerate(self.layers):
            x, c = layer(x, c)

            multi_scale_feats.append(x)
            multi_scale_coords.append(c)

        return multi_scale_feats, multi_scale_coords


class TransformerLightField(nn.Module):
    def __init__(self, latent_ch=128):
        super().__init__()

        self.num_scales = 2
        self.img_encoder = MultiscaleImgEncoder(mid_ch=latent_ch, sidelength=64)

        self.rgb_out = nn.Sequential(
            nn.GELU(),
            nn.Linear(latent_ch, 3)
        )

        self.apply(init_weights)

    def forward(self, input, val=False, compute_depth=False, timing=False):
        out_dict = {}
        query = input['query']
        context = input['context']

        rgb = context['rgb'][:, 0]
        uv = context['uv'][:, 0]

        ms_feats, ms_coords = self.img_encoder(rgb, uv/64 - 1)
        intermed_feats, intermed_coords = self.intermediate_encode(ms_feats, ms_coords, context) # Multi-scale pointcloud!

        query_rays = geometry.plucker_embedding(query['cam2world'], query['uv'], query['intrinsics'])[:, 0]
        ray_feats = self.ray_decode(query_rays, intermed_feats, intermed_coords)

        rgb = self.rgb_out(ray_feats)[:, None]
        out_dict['rgb'] = rgb
        return out_dict


class GlobalTransLightField(TransformerLightField):
    def __init__(self, **kwargs):
        super().__init__()
        latent_ch = 128

        self.rgb_out = nn.Sequential(
            nn.Linear(latent_ch+6, latent_ch),
            nn.GELU(),
            nn.Linear(latent_ch, latent_ch),
            nn.GELU(),
            nn.Linear(latent_ch, 3)
        )

    def intermediate_encode(self, feats, coords, context):
        return feats, coords

    def ray_decode(self, query_rays, feats, coords):
        n = query_rays.shape[1]
        z = feats[-1].squeeze(-1).squeeze(-1).unsqueeze(1)
        z = repeat(z, 'b n ch -> b (n n1) ch', n1=n)  # (b, h*w/ks**2, 1, self.nh*self.hd)
        return torch.cat((z, query_rays), dim=-1)


class Multiscale2DLightField(TransformerLightField):
    def __init__(self, latent_ch=128):
        super().__init__()

        self.ray_decoder = MultiScaleDecoder(channels=latent_ch, patched=False, scale_idcs=[10, 14, 17, 18], coord_dim=6,
        # self.ray_decoder = MultiScaleDecoder(channels=latent_ch, patched=False, scale_idcs=[14, 16, 17, 18], coord_dim=6,
        # self.ray_decoder = MultiScaleDecoder(channels=latent_ch, patched=False, scale_idcs=[13, 16, 17, 18], coord_dim=6,
                                             concat_query_coords=True, pos_encoding=True)
        self.coord_map = nn.Linear(latent_ch, 3, bias=True)

        self.rgb_out = nn.Sequential(
            nn.GELU(),
            nn.Linear(latent_ch, 3)
        )
        self.apply(init_weights)

    def intermediate_encode(self, feats, coords, context):
        return feats, coords

    def ray_decode(self, query_rays, feats, coords):
        return self.ray_decoder(feats, query_rays)[-1]


class MultiscalePointLightField(TransformerLightField):
    def __init__(self, latent_ch=128):
        super().__init__()

        self.scale_idcs = [10, 14, 17, 18]

        self.pc_decoder = MultiScaleDecoder(channels=latent_ch, patched=True, scale_idcs=self.scale_idcs,
                                            coord_dim=2, concat_query_coords=True)
        self.depth_map = nn.Linear(latent_ch, 1, bias=True)

        self.ray_decoder = nn.ModuleList([
            GlobalAttention(128, query_dim=128)
        ])

        self.init_proj = nn.Linear(6, latent_ch)

        self.rgb_out = nn.Sequential(
            nn.GELU(),
            nn.Linear(latent_ch, 3)
        )
        self.apply(init_weights)

    def intermediate_encode(self, feats, coords, context):
        query_coords = rearrange(coords[0], 'b ch h w -> b (h w) ch')
        pc_feats = self.pc_decoder(feats, query_coords)

        pc_coords = []
        for x in pc_feats:
            depth = torch.square(self.depth_map(x))
            pc_coords.append(
                geometry.world_from_xy_depth(query_coords, depth, context['cam2world'][:, 0], context['intrinsics'][:, 0])
            )
        return pc_feats, pc_coords

    def ray_decode(self, query_rays, feats, coords):
        '''feats: list of 2D features.'''

        feats_concat = []
        for decoder, feat, coord in zip(self.ray_decoder, feats, coords):
            # Compute distance of every ray to 3D coordinate
            dist_mat, local_coords = geometry.plucker_to_point(query_rays, coords)
            knn_idx = dist_mat.argsort()[:, :, :9] # k points nearest to the lines

            knn_coords = util.index_points(local_coords, knn_idx)
            knn_feats = util.index_points(feats, knn_idx)

            feats_concat.append(decoder(knn_feats, knn_coords))

        return self.ray_decoder(feats, query_rays)[-1]
