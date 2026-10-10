# 팀 저장소 복사 및 공유 점검

## 원격 main 연결과 문서 통합

최초 복사 이후 사용자가 GitHub 인증과 로컬 HEAD를 정상화했다. 원격 main은 `842362b9b1f5fbfdb838ed0cd392fc812ed069a8`이며 기존 커밋 5개, 파일 5개(README, .gitignore, docs/src/tests의 .gitkeep)를 확인했다. 로컬 main을 이 커밋에 연결하고 origin/main을 upstream으로 지정했다. 최초 fetch에 추적 refspec이 없어 FETCH_HEAD만 갱신됐으므로 origin/main을 명시적으로 가져오고 표준 remote.origin.fetch 매핑을 추가했다.

충돌은 루트 README.md와 .gitignore 두 개였다. 준비한 파일 86개 전체, 충돌 원본, 원격 main 파일 ZIP을 저장소 밖 고유 백업 폴더에 보존했다. 백업 위치와 원격 커밋은 루트 SHARE_MANIFEST.json의 remote_integration 항목에 기록했다.

원격 README의 프로젝트 소개·목표를 보존하고 재구성 모듈 실행 안내를 덧붙였다. 원격 ignore 규칙도 유지하고 `*.csv`에서 선별 근거 폴더만 예외로 허용했다. 데이터·가중치·캐시는 계속 제외한다. 공유 코드·대표 결과 내용은 변경하지 않았다. 원격 docs/src/tests 파일도 유지한다.

git add·commit·push, reset --hard, clean, 추가 학습·모델 추론은 실행하지 않았다. checkout으로 원격의 기존 파일을 가져왔으며, 공유 변경 사항은 미추적 또는 unstaged 상태로 남겼다. 아래 최초 복사 기록의 HEAD 오류 및 추적 파일 0개 설명은 당시 상태이며 현재 상태는 이 절을 따른다.


2026-10-11. 원본 `ecg-capstone/`은 읽기 전용으로 취급하고, 원본 `SHARE_FILES.txt`의 82개 파일을 팀 저장소 `ecg_reconstruction/`에 복사했다. 저장소 루트에 안내 README, `.gitignore`, 전체 파일 목록과 원본 대비 해시 목록을 추가했다. 이 폴더의 README·실험 요약·인터페이스·이 점검 문서는 새 위치와 제외 파일 안내를 위해 복사본만 수정했다.

## 현재 팀 저장소 검사

- Python 코드 23개와 테스트 1개의 내용·SHA-256을 원본과 비교한다. 로컬 import에 필요한 파일과 데이터 준비 코드를 모두 포함한다. 소스 해시 검증 로직을 수정하거나 우회하지 않는다.
- 대표 결과 45개와 결과 README·MANIFEST를 포함한다. CSV·JSON·PNG와 NPZ 규격 JSON은 원본 공유 목록의 바이트를 유지한다. 원본 파형 NPZ·NPY, WFDB, 가중치, 가상환경, 캐시와 전체 중복 결과는 복사하지 않는다.
- 공유 문서의 상대 링크가 저장소 안의 포함 파일로 연결되는지 확인한다. 제외한 실행 입력·출력·체크포인트의 경로는 실행 이력 또는 별도 준비 대상으로 표시한다. [전달 안내](docs/CHECKPOINT_HANDOFF.md)에 데이터 해시, 여섯 체크포인트 경로·해시와 이력 복원 방법이 있다.
- GitHub/AWS/API 토큰, private key, 자격증명 포함 URL, 문자열 secret 패턴과 민감 파일명을 검사한다. 검사 결과에는 비밀정보 값 자체를 출력하지 않는다. 패턴 검사만으로 모든 비밀정보의 부재를 보장하지 않는다.
- Git의 실제 ignore 규칙과 `text` attribute를 읽기 검사한다. `.gitignore`는 이미 추적된 파일을 해제하지 않으므로 `git ls-files -ci --exclude-standard`도 확인한다.
- 현재 `git status --porcelain=v1 --untracked-files=all`과 `git ls-files`는 실행 가능하며 복사 전 출력은 모두 비어 있었다. `.git/HEAD`는 `ref: refs/heads/.invalid`여서 브랜치·커밋 조회는 실패했다. Git 내부 설정은 고치지 않았다. commit이 없다는 것은 사용자 제공 정보이며 `git log`로 확인할 수 없었다.
- Git 설치·저장소 초기화·연결·`git add`·commit·push, 의존성 설치, 데이터 다운로드, 학습·모델 추론을 실행하지 않았다.

저장소 루트 기준 전체 복사·추가 파일 목록은 [SHARE_FILES.txt](../SHARE_FILES.txt), 파일별 원본·공유본 SHA-256은 [SHARE_MANIFEST.json](../SHARE_MANIFEST.json)에 있다. 해시 목록 자체는 자기 참조를 피하려고 자체 해시를 포함하지 않는다. 최종 파일 수·용량·읽기 검사 결과는 해당 해시 목록과 작업 완료 보고를 참고한다.

## 원본에서 작성한 이전 점검 기록

아래는 Git 설치 전 원본 폴더에서 작성한 과거 기록이다. Git 미설정에 관한 문장은 당시 원본 상태를 설명하며, 현재 팀 저장소의 상태는 위 절을 따른다. 당시 공유 후보 목록과 실제 Git index는 구분된다.

점검일: 2026-10-11. 현재 파일을 읽고 문서·공유 설정·선별 결과 사본만 정리했다. 사용자 확인대로 Git/GitHub 설정과 별도 팀 업로드 규칙은 아직 없다. Git 설치, 저장소 초기화·연결, staging, commit, push는 수행하지 않았다.

## Git 상태와 이미 추적된 파일

`git status --short --untracked-files=all`을 시도했으나 Git 명령을 찾지 못했다. 프로젝트와 모든 상위 경로에 `.git` 파일/폴더도 없다. 기존 `.gitignore`, 의존성 파일, AGENTS.md, CONTRIBUTING, CODEOWNERS 등 팀 규칙 파일은 발견하지 못했다.

따라서 현재 작업 폴더에는 확인 가능한 Git index·staging 목록이 없다. `SHARE_FILES.txt`는 실제 존재하는 파일과 작성한 제외 규칙을 기준으로 만든 **공유 대상 목록**이며 `git ls-files` 결과가 아니다. 다른 저장소에서 이미 추적 중인 파일은 확인할 수 없다.

`.gitignore`는 이미 추적 중인 파일을 제외하지 않는다. 나중에 실제 팀 저장소에 복사하거나 Git을 설정하면 아래 읽기 전용 명령으로 다시 확인한다. 이번 작업에서는 실행하지 않았다.

```powershell
git status --short --untracked-files=all
git ls-files
git ls-files -ci --exclude-standard
git diff --cached --name-only
git check-ignore -v data/processed/ptbxl_100hz_1000records.npz .venv/pyvenv.cfg checkpoints/best_model.pth
```

`git ls-files -ci --exclude-standard`에 데이터·체크포인트·비밀 파일이 나오면 이미 추적된 파일이므로 별도 index 정리가 필요하다. 이 점검에서는 파일 삭제나 추적 해제 명령을 실행하지 않았다. 실제 index가 생기기 전에는 업로드 대상을 확정했다고 표현하지 않는다.

## 공유 대상

- 기존 Python 실험·검증·추론 코드 전체와 기존 unittest 파일. 체크포인트 소스 해시 검증을 보존하기 위해 `.py`는 수정하지 않았다.
- README, 실험 요약, 입출력 규격, 환경·전달·점검 문서.
- `.gitignore`, `.gitattributes`, `requirements.txt`, `SHARE_FILES.txt`.
- `reports/representative/`의 대표 CSV·JSON·PNG와 원본/사본 SHA-256 목록.

정확한 전체 파일 경로는 [SHARE_FILES.txt](SHARE_FILES.txt)에 한 줄씩 기록했다. 원본 데이터와 가중치가 없어도 README·요약·대표 결과는 읽을 수 있다. 기존 체크포인트로 실행하려면 [전달 안내](docs/CHECKPOINT_HANDOFF.md)에 적힌 별도 파일을 준비해야 한다.

## 제외 대상과 실제 점검 범위

| 항목 | 처리 |
|---|---|
| `data/` 전체: 원본 WFDB·metadata·가공 NPZ | 루트 경로를 ignore, 로컬 보존 |
| `.venv/`, Python·도구 캐시 | ignore, 로컬 보존 |
| `checkpoints/`, `*.pth` 등 가중치 | ignore, 별도 전달 |
| 원본 `results/` 전체와 중복 결과 | ignore, 대표 사본만 공유 |
| 원본 `examples/`의 NPY/NPZ 예제 | ignore, 출처·규격 JSON 사본만 공유 |
| 루트 `ecg_12lead.png` 등 초기 중복 출력 | ignore, 로컬 보존 |
| `.env*`, private key, credentials, 에디터 설정 | ignore |

대표 PNG는 팀 공유할 기존 결과를 명시적으로 선별했다. 새 waveform이나 test 사례를 생성하지 않았다. 기존 의미가 모호했던 `selected_record_metrics.csv` 대신 수정 CSV를 공유한다. output NPZ 대신 shape·dtype·Lead 순서만 담은 JSON을 공유한다.

기존 `.py`·문서·tests 및 결과의 텍스트 파일을 대상으로 GitHub/AWS/API 토큰, private key, 자격증명 URL, 문자열에 직접 넣은 secret 패턴을 검사했고, 최종 공유 대상에도 다시 적용했다. 해당 패턴과 비밀정보 의심 파일명은 발견하지 못했다. 자동 패턴 검사만으로 모든 비밀정보의 부재를 보장하지는 않는다. 제외된 가상환경과 binary 체크포인트를 비밀정보 검사 완료 대상으로 간주하지 않는다.

결과 JSON과 일부 검사 CSV에는 로컬 사용자 경로가 있어 공유 사본에서만 상대 경로로 바꿨다. 원본은 그대로 보존했으며 원본·사본 해시는 `reports/representative/MANIFEST.json`으로 구분한다. 지표 수치는 바꾸지 않았다. 저장된 PTB-XL ecg_id/patient_id는 실험 추적을 위해 공유 근거에 남겼다.

## 재현성과 검사의 한계

Python 및 대표 결과의 Git 줄바꿈 변환을 `.gitattributes`로 막았다. 기존 소스 해시 검증을 우회하거나 제거하지 않았다. Git이 없으므로 실제 Git 엔진의 ignore/attributes 적용 검사는 아직 할 수 없었고, 작성한 단순 규칙에 대한 정적 파일 분류로 공유 목록을 만들었다.

직접 import하는 다섯 외부 패키지의 버전을 현재 설치 메타데이터에서 확인해 `requirements.txt`에 기록했다. 새 가상환경에서의 설치·다운로드는 실행하지 않았으며 완전한 transitive lock도 만들지 않았다. CPU/CUDA 빌드와 장치가 바뀌면 결과의 bitwise 재현성은 보장하지 않는다.

문서 링크, 대표 결과 해시와 수치, 공유 후보의 비밀정보 패턴 및 데이터·원본 결과·체크포인트·Python 소스 보존 여부를 검사했다. 학습·모델 추론은 재실행하지 않았다. 저장소 공개 범위와 데이터 이용 조건은 이후 팀에서 결정한다.
