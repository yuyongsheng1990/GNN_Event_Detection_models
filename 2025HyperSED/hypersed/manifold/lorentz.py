import geoopt
import torch
import geoopt.manifolds.lorentz.math as lmath


class Lorentz(geoopt.Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super(Lorentz, self).__init__(k, learnable)

    def cinner(self, x, y):  # 计算lorentz双曲内积。
        x_clone = x.clone()
        x_clone.narrow(-1, 0, 1).mul_(-1)  # 取x第1列，并取反。注意：这里x0分量通常对应维度，而时间分量x0具有相反符号，因此需要取反。
        return x_clone @ y.transpose(-1, -2)

    def to_poincare(self, x, dim=-1):
        x = x.to(self.device)
        dn = x.size(dim) - 1
        return x.narrow(dim, 1, dn) / (x.narrow(dim, 0, 1) + torch.sqrt(self.k))

    def from_poincare(self, x, dim=-1, eps=1e-6):
        x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
        res = (
                torch.sqrt(self.k)
                * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
                / (1.0 - x_norm_square + eps)
        )
        return res

    def Frechet_mean(self, x, weights=None, keepdim=False):
        if weights is None:
            z = torch.sum(x, dim=0, keepdim=True)  # (1,3)
        else:
            z = torch.sum(x * weights, dim=0, keepdim=keepdim)
        denorm = self.inner(None, z, keepdim=keepdim)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        z = z / denorm
        return z
        # no square distances.
        # no use geodesic optimization
        # no multiple ierations as Frechet mean requires.