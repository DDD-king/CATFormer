import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from models import CATFormer_tiny, CATFormer_small, CATFormer_base, CATFormer_large, CATFormer_huge

#model = CATFormer_tiny()
model = CATFormer_small()
#model = CATFormer_base()
#model = CATFormer_large()
# model = CATFormer_huge()
model.eval()
print(model)
flops = FlopCountAnalysis(model, torch.rand(1, 3, 224, 224))
print(flop_count_table(flops))