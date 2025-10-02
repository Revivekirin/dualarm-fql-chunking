python - <<'PY'
import gymnasium as gym, importlib
importlib.import_module("gym_aloha")
e = gym.make("gym_aloha/AlohaInsertion-v0")
print("obs keys:", list(e.observation_space.spaces.keys()))
for k, sp in e.observation_space.spaces.items():
    try: shp = sp.shape
    except: shp = None
    print(f" - {k}: shape={shp}")
PY
