# MADiff for Robotics

This tutorial integrates a MADiff hand-trajectory forecast into an imitation-learning policy as a conditioning token, using [ACT](https://github.com/tonyzhaozh/act) with one RGB-only (no depth) head camera and one wrist camera as the worked example.

MADiff predicts the future hand path in the head-camera image from the current frame, and that forecast is fed to the policy as one extra token, giving it a scene-conditioned prior for where the action is heading. 

The pipeline has four stages: train the forecaster on human-hand demonstrations, generate one forecast per robot demonstration, inject it into the policy, and deploy.

## Data

Prepare two datasets of the same task from the same head viewpoint:

- Human-hand demonstrations for the forecaster: Head-camera video, extract 2D hand keypoints with HaMeR or WiLoR into the EgoPAT3D layout, tracking `J` points, e.g. joints (`0`, `4`, `12`).
- Robot demonstrations for the policy

*Note: The default MADiff backends use depth. For a RGB-only camera, replace the 3D occupancy memory with a 2D scene memory, e.g. the first-frame DINOv2 patch tokens. Extract them with:*

```python
proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
patch = model(**proc(images=frame0, return_tensors="pt").to("cuda"))[0][0, 1:, :]  # (256, 768)
```

## Train the Forecaster

```bash
python run_train.py --dataset_backend egopat3d --extra_args "--epochs 100 --lr 1e-4"
```

The forecaster outputs the normalized `(u, v)` of each tracked joint at each future step, with shape `(T, J, 2)`.

## Generate Forecasts

At observation step 0 the only surviving conditioning is the first frame, so each robot episode needs only its first frame. 

For each episode, compute the first-frame DINOv2 patch tokens, run `p_sample_loop` over `T` noised steps, stack the per-joint `(T, 2)` tracks into `(T, J, 2)`. Keep this forecast constant across the whole episode for global planning.

## Inject into ACT

Resample the forecast to a fixed length and feed it as a flat vector. For example, take `(T, J, 2)`, resample to `K = 16`, subtract `0.5` to center it, and flatten to `K * J * 2`, which is 96 for `J = 3`.

Return the forecast from the dataset as one extra item:

```python
traj = np.load(f"forecasts/episode_{ep}.npy")                                 # (T, 3, 2) in [0, 1]
hand_traj = torch.from_numpy((resample(traj, 16) - 0.5).reshape(-1)).float()  # (96,)
```

Add one token beside ACT's latent and proprioception tokens in `detr/models/detr_vae.py`:

```python
self.additional_pos_embed = nn.Embedding(3, hidden_dim)   # was 2
self.hand_proj = nn.Linear(K * J * 2, hidden_dim)
self.hand_null = nn.Parameter(torch.zeros(hidden_dim))    # learned null token

hand_tok = self.hand_proj(hand_traj)
if self.training:                                         # conditioning dropout, p = 0.3
    keep = (torch.rand(bs, 1, device=hand_tok.device) >= 0.3).float()
    hand_tok = keep * hand_tok + (1 - keep) * self.hand_null
```

Stack it into the decoder cross-attention input in `detr/models/transformer.py`:

```python
addition_input = torch.stack([latent_input, proprio_input, hand_tok], axis=0)  # was 2 rows
```

## Deploy

Compute the forecast once at t=0, hold it constant for the whole rollout, and pass it to the policy at every step. On our AgiBot G01 pick-and-place task, the forecast widened the reliable grasp region and stabilized long carries after a grasp.
