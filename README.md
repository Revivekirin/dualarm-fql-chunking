# How to Start

## 1. Clone repositories

```bash
# Main project
git clone https://github.com/Revivekirin/dualarm-fql-chunking.git
cd dualarm-fql-chunking

# ALOHA environment
git clone https://github.com/huggingface/gym-aloha.git
cd gym-aloha
pip install -e .
```

---

## 2. Modify gym-aloha defaults

`gym_aloha/__init__.py` 수정:

```python
# Before
kwargs = {"obs_type": "pixels", "task": "insertion"}

# After
kwargs = {"obs_type": "pixels_agent_pos", "task": "insertion"}
```

---

## 3. Download datasets

```bash
# insertion dataset
git clone https://huggingface.co/datasets/lerobot/aloha_sim_insertion_scripted_image

# transfer dataset
git clone https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_scripted_image
```

---

## 4. Environment variables

데이터셋 위치를 환경변수로 등록합니다:

```bash
# For insertion task
export ALOHA_DATASET_DIR=/home/<USER>/workspace2/dualarm-fql-chunking/gym-aloha/aloha_sim_insertion_scripted_image/

# For transfer task
export ALOHA_DATASET_DIR=/home/<USER>/workspace2/dualarm-fql-chunking/gym-aloha/aloha_sim_transfer_cube_scripted_image/
```

---

## 5. Install Git LFS

ALOHA 데이터셋은 Git LFS 기반으로 관리됩니다.
conda 환경에서 설치:

```bash
# fql conda env 내부
conda install -c conda-forge git-lfs
```

대안:

* micromamba/mamba 사용
* GitHub 릴리스에서 tar.gz 다운로드 → `~/.local/bin`에 설치 후 PATH 추가

설치 확인:

```bash
git lfs version
```

---

## 6. Initialize Git LFS

```bash
git lfs install
```

---

## 7. Fetch dataset binaries

데이터셋 디렉토리로 이동 후:

```bash
cd /home/<USER>/workspace2/dualarm-fql-chunking/gym-aloha/aloha_sim_insertion_scripted_image

# Fetch all LFS objects
git lfs fetch --all

# Replace pointer files with real binaries
git lfs checkout
# or
git lfs pull
```

## 8. script 실행
```bash
./scripts/run_fql.sh
```