
import wfdb
import numpy as np
import matplotlib.pyplot as plt

print("ECG 데이터를 불러오는 중...")

# PTB-XL 데이터 불러오기
record = wfdb.rdrecord(
    "00001_lr",
    pn_dir="ptb-xl/1.0.3/records100/00000"
)

# 심전도 데이터 확인
signals = record.p_signal
fs = record.fs
lead_names = record.sig_name

print("데이터 크기:", signals.shape)
print("샘플링 주파수:", fs, "Hz")
print("Lead 목록:", lead_names)

# 시간축 만들기
time = np.arange(len(signals)) / fs

# 12개 심전도 그래프 그리기
fig, axes = plt.subplots(6, 2, figsize=(14, 12))
axes = axes.flatten()

for i in range(12):
    axes[i].plot(time, signals[:, i])
    axes[i].set_title("Lead " + lead_names[i])
    axes[i].set_ylabel("mV")
    axes[i].grid(True)

fig.suptitle("PTB-XL 12-Lead ECG")
fig.supxlabel("Time (seconds)")

plt.tight_layout()

# 이미지 저장
plt.savefig("ecg_12lead.png", dpi=300)

# 그래프 화면에 표시
plt.show()



# ==================================
# PART 1: AI 학습용 데이터 분리
# ==================================

# 1. Lead I을 입력 데이터 X로 사용
lead_i_idx = lead_names.index("I")
X = signals[:, lead_i_idx:lead_i_idx + 1]

# 2. Lead I을 제외한 나머지 11개를 정답 Y로 사용
target_indices = [
    i for i, name in enumerate(lead_names)
    if name != "I"
]

Y = signals[:, target_indices]

# 3. 데이터 확인
print("\n===== AI 학습 데이터 확인 =====")
print("입력 X 크기:", X.shape)
print("정답 Y 크기:", Y.shape)
print("입력 Lead:", lead_names[lead_i_idx])
print("정답 Lead:", [lead_names[i] for i in target_indices])

# 4. NaN, 무한대 검사
print("X 정상값 여부:", np.isfinite(X).all())
print("Y 정상값 여부:", np.isfinite(Y).all())

# 5. 데이터 크기 자동 검증
assert X.shape == (1000, 1), "X 데이터 크기 오류"
assert Y.shape == (1000, 11), "Y 데이터 크기 오류"
assert np.isfinite(X).all(), "X에 비정상값 존재"
assert np.isfinite(Y).all(), "Y에 비정상값 존재"

print("데이터 검증 완료!")
