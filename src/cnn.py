import torch
import torch.nn as nn

class DualChannelCSI_CRNN(nn.Module):
    """
    CRNN Architecture for Wi-Fi CSI Human Activity Recognition.
    
    Input shape:  (Batch, 2, 500, 242) -> [Batch, Channels (Amp+Phase), Time_Steps, Subcarriers]
    Output shape: (Batch, 20)           -> Logits for 20 activity classes (A to T)
    """
    def __init__(self, num_classes: int = 20, hidden_dim: int = 128, num_rnn_layers: int = 2):
        super(DualChannelCSI_CRNN, self).__init__()
        
        # -----------------------------------------------------------------
        # 1. Spatial Feature Extractor (2D CNN)
        # -----------------------------------------------------------------
        # Block 1: Input (B, 2, 500, 242) -> Output (B, 32, 250, 60)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # Block 2: Input (B, 32, 250, 60) -> Output (B, 64, 125, 15)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # Block 3: Input (B, 64, 125, 15) -> Output (B, 128, 125, 15)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 128 feature channels * 15 remaining subcarriers = 1920 feature dimension per time step
        self.rnn_input_dim = 128 * 15
        
        # -----------------------------------------------------------------
        # 2. Temporal Sequence Extractor (Bidirectional GRU)
        # -----------------------------------------------------------------
        self.bigru = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_rnn_layers > 1 else 0.0
        )
        
        # -----------------------------------------------------------------
        # 3. Classification Head
        # -----------------------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 2, 500, 242)
        B, C, T, S = x.shape
        
        # 1. Pass through Conv layers
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)  # Shape: (B, 128, T_reduced=125, S_reduced=15)
        
        # 2. Reshape for GRU: (B, Time_Steps, Channels * Subcarriers)
        out = out.permute(0, 2, 1, 3).contiguous()  # (B, 125, 128, 15)
        out = out.view(B, out.size(1), -1)          # (B, 125, 1920)
        
        # 3. Forward through BiGRU
        gru_out, _ = self.bigru(out)                # (B, 125, 256)
        
        # 4. Global Average Pooling across time steps
        context = torch.mean(gru_out, dim=1)        # (B, 256)
        
        # 5. Output Logits
        logits = self.fc(context)                   # (B, 20)
        return logits


# --- Quick Integrity Verification ---
if __name__ == "__main__":
    model = DualChannelCSI_CRNN(num_classes=20)
    
    # Test batch matching your preprocessed loader dimensions
    test_tensor = torch.randn(8, 2, 500, 242)
    output_logits = model(test_tensor)
    
    print("✓ Model instantiated successfully!")
    print(f"✓ Input Shape:  {test_tensor.shape}")
    print(f"✓ Output Shape: {output_logits.shape} (Expected: [8, 20])")
    print(f"✓ Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")