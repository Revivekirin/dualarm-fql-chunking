import pickle
import pprint
import numpy as np

def describe(obj, indent=0):
    pad = ' ' * indent
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            result[k] = describe(v, indent + 2)
        return result
    elif isinstance(obj, list):
        return [describe(v, indent + 2) for v in obj[:]] + (['...'] if len(obj) > 3 else [])
    elif isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape}, dtype={obj.dtype}>"
    else:
        return obj


# # 파일 경로
pkl_path = '/home/sophia435256/workspace2/dualarm-fql-chunking/exp/fql/Debug/sd000_s_22777.0.20251004_164957/params_1000000.pkl'
out_path = '/home/sophia435256/workspace2/dualarm-fql-chunking/exp/fql/Debug/sd000_s_22777.0.20251004_164957/params_1000000.txt'

# 로드
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# ndarray 요약 포함한 구조 저장
summary = describe(data)

# 저장
with open(out_path, 'w', encoding='utf-8') as f:
    pprint.pprint(summary, stream=f, width=120)

print(f"✅ 요약 저장 완료: {out_path}")

