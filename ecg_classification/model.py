
import torch
import torch.nn as nn


class ECGModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(64, 11, kernel_size=1)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ECGModel().to(device)

    # 테스트용 ECG 입력
    x = torch.randn(2, 1, 1000, device=device)

    # 모델의 예측 출력
    y = model(x)

    print("사용 장치:", device)
    print("입력 크기:", x.shape)
    print("출력 크기:", y.shape)
