import torch
from backbone import build_backbone
from util.misc import NestedTensor
from transformer import build_transformer
from detr import DETR
if __name__ == "__main__":
    x = NestedTensor(torch.randn(3, 3, 224, 224), torch.randint(0, 2, (3, 224, 224)))
    backbone = build_backbone()
    feat, pos = backbone(x)
    src, mask = feat[-1].decompose()
    print(f"pos shape -> {pos[-1].shape}")
    print(f"src shape -> {src.shape}")
    print(f"mask shape -> {mask.shape}")
    con = torch.nn.Conv2d(2048, 256, kernel_size=1)
    print(f"proj shape -> {con(src).shape}")
    print('-----------------')
    transformer = build_transformer()
    out = transformer(con(src), mask, torch.nn.Embedding(100, 256).weight, pos[-1])
    print(f"decoder out shape -> {out[0].shape}")
    print(f"encoder out shape -> {out[1].shape}")
    model = DETR(
        backbone,
        transformer,
        num_classes=91,
        num_queries=100,
        aux_loss=False,
    )
    output = model(x)
    print(f"predict logits shape -> {output['pred_logits'].shape}, query class")
    print(f"predict box shape -> {output['pred_boxes'].shape}, 4 represents x_left y_up height width")
