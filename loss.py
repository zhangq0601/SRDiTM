
class SRLoss(nn.Module):
    """超分损失函数：MSE + 傅里叶变换损失 + 离散小波变换损失"""

    def __init__(self, mse_weight=1.0, fft_weight=0.1, dwt_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.fft_weight = fft_weight
        self.dwt_weight = dwt_weight
        self.mse = nn.MSELoss()

    def fft_loss(self, pred, target):
        """傅里叶变换损失（捕捉高频信息）"""
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        return F.l1_loss(pred_amp, target_amp)

    def dwt_loss(self, pred, target):
        """离散小波变换损失（捕捉多尺度纹理）"""
        # 小波基：db1，可替换为 db2/db3 等
        pred_dwt = self._dwt_2d(pred, wavelet='db1')
        target_dwt = self._dwt_2d(target, wavelet='db1')
        return F.l1_loss(pred_dwt, target_dwt)

    def _dwt_2d(self, x, wavelet='db1'):
        """2D 离散小波变换（批量处理）"""
        coeffs = pywt.wavedec2(x.cpu().detach().numpy(), wavelet=wavelet, level=1, axis=(-2, -1))
        cA, (cH, cV, cD) = coeffs
        # 转回 tensor 并拼接小波系数
        cA = torch.from_numpy(cA).to(x.device, dtype=x.dtype)
        cH = torch.from_numpy(cH).to(x.device, dtype=x.dtype)
        cV = torch.from_numpy(cV).to(x.device, dtype=x.dtype)
        cD = torch.from_numpy(cD).to(x.device, dtype=x.dtype)
        return torch.cat([cA, cH, cV, cD], dim=1)

    def forward(self, pred, target):
        """
        Args:
            pred: 模型输出（B x 3 x H x W）
            target: 真实标签（B x 3 x H x W）
        Return:
            total_loss: 加权总损失
        """
        mse_loss = self.mse(pred, target)
        fft_loss = self.fft_loss(pred, target)
        dwt_loss = self.dwt_loss(pred, target)

        total_loss = self.mse_weight * mse_loss + \
                     self.fft_weight * fft_loss + \
                     self.dwt_weight * dwt_loss
        return total_loss