# TankWar · 戰車大戰 — self-play reinforcement learning, live in your browser

A 2–4 tank free-for-all arena where every opponent is a trained DQN agent — no scripted
enemies. The whole RL stack (self-play training, replay buffer, the network itself) runs
**in your browser**: open the page, hit Start, and watch the win-rate climb past chance
while you spectate duels or take the wheel yourself. Continuous hull rotation, momentum,
ricocheting shells, destructible cover, power-ups, escalation and sudden death modernize
the original turn-of-the-tread gym environment this grew out of.

Three tabs:

- **Train** — a fleet of self-play lanes trains live, with telemetry tiles (episodes,
  ε, loss, win-rate vs random, K/D, games/s, buffer fill) and four charts. Session-only:
  training restarts from scratch on every reload.
- **Watch** — spectate 2–4 AI tanks at 0.5×–8× speed, with pause, frame-step and
  auto-restart.
- **Play** — the human seat (WASD + Space) against 1–3 AIs, free-for-all or allied,
  with a difficulty dial and an optional ≤6° aim assist.

Everything ships as one static site; GitHub Pages deploys it on every push to `main`
(`.github/workflows/deploy.yml`).

## Origin

TankWar began as the AIST1110 course project of **LAM Yiu Fung Anson** and **FONG Shi
Yuk** (upstream: [shiyukfong/AIST1110_Project](https://github.com/shiyukfong/AIST1110_Project),
origin: [anson416/tankwar](https://github.com/anson416/tankwar)): a Python gym +
pygame tank game — 1 player vs up to 4 scripted enemies, 10 discrete actions — with a
Keras DQN trainer. That code is **still in this repo, untouched**, under `TankWar/`:
the `gym-tankwar` gym environment, the `tankwar_play.py` / `tankwar_train.py` /
`tankwar_test.py` scripts, the `.h5` models in `TankWar/tankwar/models/`, and the
training-result PNGs. The `frontend/` directory is the 2026 rebuild: the same game,
rebuilt as an in-browser real-time self-play RL arena.

<details>
<summary>Original (legacy) Python instructions — preserved verbatim</summary>

# AIST1110_Project
### Project Name:
Tank War

### Group Member: 

LAM, Yiu Fung Anson

FONG, Shi Yuk


## Steps
0. `[...]` means optional argument. 
`[...|...]` means optional argument, and choose one of the two arguments only.    
1. Open your own conda environment.
2. Execute the following commands in the Terminal:
```
cd TankWar/gym-tankwar 
conda create --name tankwar python=3.10 -y 
conda activate tankwar 
pip install -e .
```  

3. Execute the following commands in the Terminal:
```
cd ../tankwar
```     
4.  To play the game, execute:
```
python tankwar_play.py -m human [-d DIFFICULTY] [-e EPISODES]
```
   To visualize the gameplay GUI in a read-only human mode, execute:
```
python tankwar_play.py -m human_rand [-d DIFFICULTY] [-e EPISODES]
```
   To expose the game in a non-GUI mode, execute:
```
python tankwar_play.py [-d DIFFICULTY] [-e EPISODES]
```


(`DIFFICULTY`: Game mode. 0 for easy mode, 1 for hard mode. Default: 0)

(`EPISODES`:Number of games to be played. Default: 1000)

(Check `README.txt` or execute:

```
python tankwar_play.py -h
```

to get all information of the command line arguments.)

5. Before training and testing the agent, execute:
```
pip install tensorflow==2.10.0
```
6. To train the agent, execute:
```
python tankwar_train.py -s SEED -d DIFFICULTY [-traine TRAIN_EPISODES | -fast]
```
(`TRAIN_EPISODES`:The number of training episodes. Default: 1000. Suggested value: 300)

(`-fast`: train the model in fast mode. Training finishes in around 20 minutes.)

7. To test the model, execute:
```
python tankwar_test.py -f FILE [-d DIFFICULTY] [-teste TEST_EPISODE]
```
(`SEED`: Seed for the random generator.)
(`TEST_EPISODES`:The number of training episodes. Default: 100.)
(`FILE`: model file name. 4 sample model is given in `/models` folder. Note that `DIFFICULTY` 
should match the specification given by the sample file name.)

</details>

## What the repo is now

| Path | What it is |
|---|---|
| `frontend/` | The browser rebuild: TypeScript engine + DQN worker + UI (this is what deploys) |
| `TankWar/` | The original AIST1110 Python gym env, scripts, models and results (untouched) |
| `.github/workflows/deploy.yml` | CI: unit → build → artifact check → Playwright smoke → GitHub Pages |
| `SPEC.md` | The rebuilt frontend's engineering contract (world model, actions, obs, rewards, networks, scheduler, protocol, tests) |

## Quick start (the browser game)

```bash
cd frontend
npm ci
npm run dev        # http://localhost:5173 — full app, trains while open
```

Other commands:

```bash
npm run test          # unit suite (engine property tests, protocol, scheduler, …)
npm run build         # typecheck + copy wasm binaries + bundle to dist/
npm run build:check   # static checks on the built artifact
npm run build:verify  # build + artifact check + Playwright smoke (what CI runs)
npm run test:smoke    # Playwright smoke against the BUILT dist/ (vite preview)
```

Deploying is automatic: a push to `main` runs the gates in
`.github/workflows/deploy.yml` and publishes `frontend/dist` to
`https://anson416.github.io/tankwar/`.

## How training works

- **18 discrete actions** drive continuous physics: `a = (throttle·3 + steer)·2 + fire`
  (throttle coast/fwd/rev × steer left/straight/right × fire no/yes). Human keys decode
  through the same table.
- **282-float observation**, own-heading reference frame: self state (16), 3 nearest
  tanks (36), 18 nearest bullets (126), 4 power-ups (24), a 16-ray lidar (80). All
  values ∈ [−1, 1]. See `SPEC.md` for the block table.
- **DQN in a Web Worker** — one worker owns every tensor: 16 lockstep self-play lanes,
  one fleet forward per tick, a 40 000-row replay ring, DDQN (double-Q) targets with a
  hard-synced target net every 1000 steps, Huber loss + Adam. Potential-based shaping
  (Ng et al. 1999) is provably policy-neutral; fire-masking applies to the behaviour
  policy only.
- **Backend ladder**: WebGPU → WebGL → WASM → CPU (tfjs 4.22.0, wasm binaries
  self-hosted). Safari workers, CI and anything else land on a working tier.
- **Session-only**: nothing persists. Reload = a fresh fleet training from scratch —
  which is the point: you can watch it re-learn in minutes.
- **Eval** every 100 episodes: 24 greedy-vs-random episodes; the train tile shows the
  win-rate against the honest 1/N chance line.

Rewards (per agent, FFA): damage dealt +2/HP · kill +50 · crate broken +5 · pickup +8 ·
survival +0.02/tick (×2 escalation, ×4 sudden death) · damage taken −1.5/HP · death −60 ·
wall contact −0.05/tick · expired bullet −0.3 · allied-fire −3/HP · timeout placement
bonus up to +40. A kill totals ≈ +250, a death ≈ −210 — shaping never dominates outcomes.

## Controls

| Key | Action |
|---|---|
| `W` / `S` (or ↑/↓) | Throttle forward / reverse |
| `A` / `D` (or ←/→) | Steer left / right (continuous, momentum) |
| `Space` | Fire |
| `Enter` | Rematch (Play tab, after a match ends) |

Play-tab options: opponents 1–3, team mode (FFA vs allied AIs), difficulty (softmax
temperature on the opponents' action choice), aim assist (snaps your hull up to 6°
toward the intercept point — visibly, so it never silently bends your shells).

## Architecture selector

The Train tab's config form switches the network without touching the game:

- **MLP (default)** — 282 → Dense 256 ReLU → Dense 256 ReLU → Dense 18 linear
  (~143k params). Fast, and the flat obs already carries the nearest-entity blocks.
- **Entity attention (opt-in)** — 26 entity tokens (self + 3 tanks + 18 bullets +
  4 power-ups) with type embeddings, d_model 64, two pre-LN blocks of 4-head
  attention (key dim 16) + FFN residual, masked mean pool → concat with the ray
  block → Dense 128 → the same shared Dense 18 head.

Apply rebuilds the network transactionally and resets training (with a warning —
that is the point of the button). Every other knob (γ, ε schedule, lanes, batch,
buffer, trainEvery, targetSync, evalEvery, episode length, tank count) lives in the
same form with client-side validation.

## Asset licenses (carried over from the original project)

All game art and audio are byte-for-byte copies from the AIST1110 project; the
verbatim per-asset records live in
[`frontend/public/assets/ATTRIBUTION.md`](frontend/public/assets/ATTRIBUTION.md).
Summary:

- **Tanks & bullets** — [craftpix freebies](https://craftpix.net/freebies/free-2d-battle-tank-game-assets/)
  ([craftpix file license](https://craftpix.net/file-licenses/)): `tank_01_A/B.png`,
  `tank_02_A/B.png`, `bullet_01.png`, `bullet_02.png`.
- **Heart (repair icon)** — [opengameart "Heart"](https://opengameart.org/content/heart-1),
  [CC0](https://creativecommons.org/publicdomain/zero/1.0/): `heart.png`.
- **Explosion sheet** — [gushh.net free game sprites — explosion 4](http://gushh.net/blog/free-game-sprites-explosion-4/):
  `explosion.png` (1024² sheet, sampled via `drawImage` sub-rects).
- **Background** — [opengameart top-down backgrounds](https://opengameart.org/content/backgrounds-topdown-games),
  [CC-BY 3.0](https://creativecommons.org/licenses/by/3.0/): `background.png`.
- **Audio** — `background_music.wav` ([opengameart "War"](https://opengameart.org/content/war), CC0);
  `tank_engine.ogg` ([opengameart engine loop](https://opengameart.org/content/engine-loop-heavy-vehicletank),
  dual [CC-BY 3.0](https://creativecommons.org/licenses/by/3.0/) / GPL-3.0);
  `cannon_fire.ogg` ([opengameart cannon fire](https://opengameart.org/content/cannon-fire), CC0);
  `explosion.wav` — © Viktor Hahn (Viktor.Hahn@web.de),
  [CC-BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/)
  ([opengameart "9 explosion sounds"](https://opengameart.org/content/9-explosion-sounds)).

## Bilingual

The UI is fully bilingual — English and 繁體中文 — with the language toggle in the
header (persisted per browser, default following the system locale). The dark/light
theme toggle sits next to it. Both are session preferences only; no gameplay data is
ever stored.

## License note

The original project files keep their original provenance; the game assets carry the
licenses listed above. No new code license is asserted by this rewrite.
