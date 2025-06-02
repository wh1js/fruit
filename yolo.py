import torch
import torch.nn as nn
import os


class YOLOv11(nn.Module):
    def __init__(self, num_classes, anchors=None, num_scales=3):
        super(YOLOv11, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors) // num_scales if anchors else 3
        self.num_scales = num_scales
        self.num_outputs = self.num_anchors * (5 + num_classes)  # 5: x, y, w, h, confidence

        # Default anchors for 3 scales (example values, adjust for your dataset)
        self.anchors = anchors if anchors else [
            [(10, 13), (16, 30), (33, 23)],  # Scale 1
            [(30, 61), (62, 45), (59, 119)],  # Scale 2
            [(116, 90), (156, 198), (373, 326)]  # Scale 3
        ]

        # Backbone - Darknet-53 inspired
        self.backbone = self._darknet53()

        # YOLO heads for multiple scales
        self.head_s = self._yolo_head(512)  # Small objects
        self.head_m = self._yolo_head(256)  # Medium objects
        self.head_l = self._yolo_head(128)  # Large objects

        # Upsampling for feature pyramid
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _darknet53(self):
        layers = []
        # Initial conv block
        layers += self._conv_block(3, 32, 3, 1)
        layers += self._conv_block(32, 64, 3, 2)

        # Residual blocks
        layers += self._residual_block(64, 32, 1)
        layers += self._conv_block(64, 128, 3, 2)
        layers += self._residual_block(128, 64, 2)
        layers += self._conv_block(128, 256, 3, 2)
        layers += self._residual_block(256, 128, 8)
        layers += self._conv_block(256, 512, 3, 2)
        layers += self._residual_block(512, 256, 8)
        layers += self._conv_block(512, 1024, 3, 2)
        layers += self._residual_block(1024, 512, 4)

        return nn.Sequential(*layers)

    def _conv_block(self, in_channels, out_channels, kernel_size, stride):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        ]

    def _residual_block(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers += [
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_channels, in_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(0.1)
            ]
        return layers

    def _yolo_head(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels // 2, self.num_outputs, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Backbone
        features = self.backbone(x)

        # Feature pyramid network
        out_l = self.head_l(features)  # Large scale detection

        # Upsample and combine for medium scale
        features_m = self.upsample(features)
        out_m = self.head_m(features_m)

        # Upsample again for small scale
        features_s = self.upsample(features_m)
        out_s = self.head_s(features_s)

        return out_s, out_m, out_l

    def save_weights(self, path):
        """Save model weights to a .pth file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'anchors': self.anchors
        }, path)
        print(f"Model weights saved to {path}")

    def load_weights(self, path, device='cpu'):
        """Load model weights from a .pth file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found")

        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.num_classes = checkpoint['num_classes']
        self.anchors = checkpoint['anchors']
        print(f"Model weights loaded from {path}")
        return self


if __name__ == "__main__":
    # Example usage
    model = YOLOv11(num_classes=80)
    print(model)

    # Create a sample input
    x = torch.randn(1, 3, 416, 416)

    # Forward pass
    out_s, out_m, out_l = model(x)
    print(f"Small scale output shape: {out_s.shape}")
    print(f"Medium scale output shape: {out_m.shape}")
    print(f"Large scale output shape: {out_l.shape}")

    # Save model weights
    model.save_weights("yolov11_weights.pth")

    # Load model weights
    model.load_weights("yolov11_weights.pth")