import torch
import torch.nn as nn
import pywt
import torch.fft as fft

class FFTLoss(nn.Module):
    """傅里叶变换损失"""
    def forward(self, pred, target):
        pred_fft = fft.rfft2(pred, dim=(-2, -1))
        target_fft = fft.rfft2(target, dim=(-2, -1))
        loss_real = nn.functional.mse_loss(pred_fft.real, target_fft.real)
        loss_imag = nn.functional.mse_loss(pred_fft.imag, target_fft.imag)
        return loss_real + loss_imag

class DWTLoss(nn.Module):
    """离散小波变换损失"""
    def __init__(self, wavelet='haar', level=2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, pred, target):
        pred_dwt = self._dwt2(pred, self.wavelet, self.level)
        target_dwt = self._dwt2(target, self.wavelet, self.level)
        loss = 0.
        for p, t in zip(pred_dwt, target_dwt):
            loss += nn.functional.mse_loss(p, t)
        return loss / len(pred_dwt)

    def _dwt2(self, x, wavelet, level):
        coeffs = pywt.wavedec2(x.cpu().numpy(), wavelet=wavelet, level=level, axis=(-2, -1))
        coeffs_tensor = []
        for c in coeffs:
            if isinstance(c, tuple):
                for cc in c:
                    coeffs_tensor.append(torch.from_numpy(cc).to(x.device))
            else:
                coeffs_tensor.append(torch.from_numpy(c).to(x.device))
        return coeffs_tensor

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, fft_weight=0.4, dwt_weight=0.1, wavelet='haar', dwt_level=2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.fft_loss = FFTLoss()
        self.dwt_loss = DWTLoss(wavelet=wavelet, level=dwt_level)
        self.mse_weight = mse_weight
        self.fft_weight = fft_weight
        self.dwt_weight = dwt_weight

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_fft = self.fft_loss(pred, target)
        loss_dwt = self.dwt_loss(pred, target)
        total_loss = self.mse_weight * loss_mse + self.fft_weight * loss_fft + self.dwt_weight * loss_dwt
        return total_loss