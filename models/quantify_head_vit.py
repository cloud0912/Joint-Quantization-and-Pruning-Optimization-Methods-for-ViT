

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.autograd import Variable
import math
from collections import defaultdict

MIN_NUM_PATCHES = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_model_params_set(layers=12, heads=12):
    params_set = set()
    for layer in range(layers):
        for head in range(heads):
            params_set.update({
                f"mha{layer}_head_{head}_q",
                f"mha{layer}_head_{head}_k",
                f"mha{layer}_head_{head}_v",
                f"mha{layer}_head_{head}_attn",
                f"mha{layer}_head_{head}_out",
            })
        params_set.update({
            f"ffn{layer}_input1",
            f"ffn{layer}_input2",
        })
    params_set.add("classifier")
    return params_set

# Now let's generate the parameters set for a 12-layer 12-head model
keys = generate_model_params_set()

## 量化相关代码    
class EMA_Activation:
    def __init__(self, mu=0.9):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = [val[0].clone(), val[1].clone()]
        ## register 方法将传入的最小和最大值克隆并存储在 shadow 字典中，键为传入的名称。

    def __call__(self, name, x, fixed_min):
        assert name in self.shadow
        if self.shadow[name][0] * self.shadow[name][1] == 0:
            if fixed_min:
                new_xmin = torch.tensor(0.)
            else:
                new_xmin = x.min()
            new_xmax = x.max()
        else:
            if fixed_min:
                new_xmin = torch.tensor(0.)
            else:
                new_xmin = (1.0 - self.mu) * x.min() + self.mu * self.shadow[name][0]
            new_xmax = (1.0 - self.mu) * x.max() + self.mu * self.shadow[name][1]
        self.shadow[name] = [new_xmin.clone(), new_xmax.clone()]
        return [new_xmin, new_xmax]


ema_activation = EMA_Activation()
for key in keys:
    ema_activation.register(key, [torch.tensor(0), torch.tensor(0)])


class EMA_Weight:
    def __init__(self, mu=0.9):
        self.mu = mu
        self.shadow = defaultdict(float)

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x, k=8):
        assert name in self.shadow
        xmax = x.max().item()
        xmin = x.min().item()
        s = (xmax - xmin) / (2 ** k - 1)
        q = torch.div(x, s, rounding_mode="floor") * s + xmin
        self.shadow[name] = q.clone()
        return q

def quantization_activations(X, name, k=8, fixed_min=False):
    if X.requires_grad:
        ema_activation(name, X, fixed_min)
    xmin, xmax = ema_activation.shadow[name]
    s = (xmax - xmin) / (2 ** k - 1)
    q = torch.div(torch.clamp(X, min=xmin, max=xmax), s, rounding_mode="trunc") * s + xmin
    return q


class Quantization_Activations(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, fixed_min, name, k=8):
        return quantization_activations(X, name, k, fixed_min)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None, None, None


_quantization_activations = Quantization_Activations.apply    
   
    
## 剪枝相关代码    
class channel_selection(nn.Module):
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        input_tensor: (B, num_patches + 1, dim). 
        """
        output = input_tensor.mul(self.indexes)
        return output

    
    
    
## 模型相关代码    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.residual = fn
    def forward(self, x, **kwargs):
        return self.residual(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, layers, dropout = 0.):
        super().__init__()
        self.layers = layers
        self.FFN1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.FFN2 = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        key_input1 = "ffn{}".format(self.layers) + "_input1"
        key_input2 = "ffn{}".format(self.layers) + "_input2"
        x = _quantization_activations(x, False, key_input1)
        
        x = self.FFN1(x)
        x = _quantization_activations(x, False, key_input2)
        x = self.FFN2(x)
        return x


class ClassifierQuant(nn.Module):
    def __init__(self, dim, mlp_dim, num_classes, dropout_classifier):
        super(ClassifierQuant, self).__init__()  
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        key_name = "classifier"
        x = _quantization_activations(x, False, key_name)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)        
        
        
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, channels = 3, emb_dropout = 0.):
        super(PatchEmbedding, self).__init__()
        
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, img, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        return x
       
        
class Attention(nn.Module):
    def __init__(self, dim, layers, heads = 8, dropout = 0.):
        super().__init__()
        
        self.layers = layers
        self.heads = heads
        self.scale = dim ** -0.5

        # self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_q = nn.Linear(dim, dim , bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        common = "mha{}".format(self.layers)
        
        b, n, _, h = *x.shape, self.heads
        ## 先进行线性映射，再进行分头操作
        # pruning   torch.Size([4, 65, 512])
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)  ## (batchsize, num_batches+1, dim)-->(batchsize, heads, num_batches+1, head_dim)
        k = self.to_k(x)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        v = self.to_v(x)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)
        
        # 量化 QKV
#         for i in range(h):
#             q[:, i, :, :] = _quantization_activations(q[:, i, :, :], False, common + "_head_" + str(i) + "_q")
#             k[:, i, :, :] = _quantization_activations(k[:, i, :, :], False, common + "_head_" + str(i) + "_k")
#             v[:, i, :, :] = _quantization_activations(v[:, i, :, :], False, common + "_head_" + str(i) + "_v")

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  ## （batchsize, heads, len_q, len_k）,其中(len_q = len_k = num_batches+1)

        # 量化 attention
        for i in range(h):
            dots[:, i, :, :] = _quantization_activations(dots[:, i, :, :], False, common + "_head_" + str(i) + "_attn")

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
            
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        # 量化输出
        for i in range(h):
            out[:, i, :, :] = _quantization_activations(out[:, i, :, :], False, common + "_head_" + str(i) + "_out")
        out = rearrange(out, 'b h n d -> b n (h d)')  ## (batchsize, heads, num_batches+1, head_dim)-->(batchsize, num_batches+1, dim)
        out =  self.to_out(out)
        
        return out

    
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout, layers = i))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout, layers = i)))
            ])) 
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x
    
    
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dropout = 0., emb_dropout = 0., k=8):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim, channels, emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.classifier = ClassifierQuant(dim, mlp_dim, num_classes, dropout)
        
        self.ema_weight = None
        self.init_params()
        self.ema_init()
        self.k = k

    def forward(self, img, mask = None):
        
        x = self.patch_embedding(img)
        x = self.transformer(x, mask)
        x = self.classifier(x)
        return x
    
    def init_params(self, default_initialization=False):
        # Not mentioned in the paper, but other implementations used xavier.
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def ema_init(self):
        self.ema_weight = EMA_Weight()
        for name, params in self.named_parameters():
            if params.requires_grad and "bias" not in name:
                self.ema_weight.register(name, params)

    def apply_ema(self, k=8):
        if not self.ema_weight:
            self.ema_init()
        for name, params in self.named_parameters():
            if params.requires_grad and "bias" not in name:
                self.ema_weight(name, params, k)

def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # setup_seed(200)
    b,c,h,w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    net = ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = 10,
        dim = 512,
        depth = 2,
        heads = 2,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    y = net(x)
    # for name, layer in net.named_children():
    #     for name1, layer1 in getattr(net, name).named_children():
    #         print(name1)

    # print(y)
#     for name, module in net.named_modules():
#         print(name)
    # print(net)
