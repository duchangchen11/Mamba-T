import pytest
import torch
from torch.utils.data import DataLoader
from scripts.eth_ucy_utils import evaluate

class ZeroRelativePrediction(torch.nn.Module):
    def forward(self,x):return {'future_pred':torch.zeros(len(x),12,2)}

def test_world_coordinate_metrics_weight_samples():
    last=torch.tensor([5.,7.]);items=[]
    for error in (0.,1.,3.):
        future=last[None].repeat(12,1)+torch.tensor([error,0.])
        items.append({'obs_input':torch.zeros(8,4),'last_obs_pos':last,'future_abs':future})
    result=evaluate(ZeroRelativePrediction(),DataLoader(items,batch_size=2),torch.device('cpu'))
    assert result['ADE']==pytest.approx(4/3)
    assert result['FDE']==pytest.approx(4/3)
