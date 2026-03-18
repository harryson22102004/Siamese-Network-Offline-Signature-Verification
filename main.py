import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
 
class SigEncoder(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        bb = models.resnet18(pretrained=False)
        self.enc = nn.Sequential(*list(bb.children())[:-1], nn.Flatten(),
                                   nn.Linear(512, emb))
    def forward(self, x): return F.normalize(self.enc(x), p=2, dim=1)
 
class SiameseSignature(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.encoder = SigEncoder(emb)
        self.distance_head = nn.Sequential(nn.Linear(emb, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x1, x2):
        e1 = self.encoder(x1); e2 = self.encoder(x2)
        diff = (e1 - e2).abs()
        return torch.sigmoid(self.distance_head(diff)).squeeze(), e1, e2
 
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__(); self.m = margin
    def forward(self, e1, e2, label):
        dist = F.pairwise_distance(e1, e2)
        return (label*dist.pow(2) + (1-label)*F.relu(self.m-dist).pow(2)).mean()
 
def evaluate_eer(model, pairs, labels):
    """Equal Error Rate evaluation."""
    model.eval()
    scores = []
    with torch.no_grad():
        for (x1, x2) in pairs:
            pred, e1, e2 = model(x1, x2)
            scores.append(pred.item())
    import numpy as np
    thresholds = np.linspace(0, 1, 100)
    scores = np.array(scores); labels = np.array(labels)
    far_rates = [((scores[labels==0] > t).sum()/(labels==0).sum()) for t in thresholds]
    frr_rates = [((scores[labels==1] < t).sum()/(labels==1).sum()) for t in thresholds]
    diff = np.abs(np.array(far_rates) - np.array(frr_rates))
    eer_idx = diff.argmin()
    return (far_rates[eer_idx]+frr_rates[eer_idx])/2, thresholds[eer_idx]
 
model = SiameseSignature(128); criterion = ContrastiveLoss()
x1 = torch.randn(8, 3, 112, 112); x2 = torch.randn(8, 3, 112, 112)
labels = torch.randint(0, 2, (8,)).float()
pred, e1, e2 = model(x1, x2)
loss = criterion(e1, e2, labels)
print(f"Contrastive loss: {loss.item():.4f}")
pairs = [(torch.randn(1,3,112,112), torch.randn(1,3,112,112)) for _ in range(20)]
eer, thresh = evaluate_eer(model, pairs, [i%2 for i in range(20)])
print(f"EER: {eer:.3f} at threshold: {thresh:.3f}")
