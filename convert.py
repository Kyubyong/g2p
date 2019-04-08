import torch
import numpy as np

_ckpt = "g2p_en/20.pt"
ckpt = torch.load(_ckpt)

enc_emb, enc_w_ih, enc_w_hh, enc_b_ih, enc_b_hh, dec_emb, dec_w_ih, dec_w_hh, dec_b_ih, dec_b_hh, fc_w, fc_b  = ckpt.values()

np.savez('g2p_en/checkpoint20.npz',
         enc_emb=enc_emb.cpu().numpy(), enc_w_ih=enc_w_ih.cpu().numpy(),
         enc_w_hh=enc_w_hh.cpu().numpy(), enc_b_ih=enc_b_ih.cpu().numpy(),
         enc_b_hh=enc_b_hh.cpu().numpy(), dec_emb=dec_emb.cpu().numpy(),
         dec_w_ih=dec_w_ih.cpu().numpy(), dec_w_hh=dec_w_hh.cpu().numpy(),
         dec_b_ih=dec_b_ih.cpu().numpy(), dec_b_hh=dec_b_hh.cpu().numpy(),
         fc_w=fc_w.cpu().numpy(), fc_b=fc_b.cpu().numpy())