// i18n.ts — bilingual UI strings (EN / 繁體中文). Tiny on purpose: a typed
// dictionary + {param} interpolation + localStorage persistence. Default
// language follows the browser; the toggle persists an explicit choice.
//
// Parity is compiler-enforced: StringKey is keyof the EN dict, and `zh` is
// annotated as Record<StringKey, string>, so a key added to one dict and not
// the other is a type error rather than a silent missing label at runtime.

export type Lang = "en" | "zh";

const STORAGE_KEY = "tankwar.lang";

export const STRINGS = {
  en: {
    // Shell
    "app.title": "TankWar · 戰車大戰",
    "app.subtitle": "self-play reinforcement learning, live in your browser",
    "tab.train": "Train",
    "tab.play": "Play",
    "tab.watch": "Watch",
    "status.loading": "Engine loading…",
    "status.ready": "Engine ready",
    "status.running": "Training",
    "header.language": "Language",
    "header.theme": "Theme",
    "header.sound": "Sound on",
    "header.soundOff": "Sound off",

    // Per-tab scaffolding notes (each tab grows real content in later
    // milestones; the placeholder tells an early user what is coming).
    "train.placeholder":
      "Training controls and live charts arrive with the learning milestone — scaffold build.",
    "play.placeholder": "Play mode arrives with the trained network — scaffold build.",
    "watch.placeholder": "Watch mode arrives with the renderer — scaffold build.",

    // Train tab — start/stop + stat tiles
    "train.start": "Start training",
    "train.stop": "Stop training",
    "train.backend": "Backend",
    "train.episodes": "Episodes",
    "train.decisions": "Decisions",
    "train.trainSteps": "Train steps",
    "train.epsilon": "ε (exploration)",
    "train.loss": "Loss (smoothed)",
    "train.avgReturn": "Avg return",
    "train.winRate": "Win rate vs random",
    // {chance} is the tank count N — the honest bar for an N-player FFA is 1/N.
    "train.winRateChance": "chance = 1/{chance}",
    "train.kd": "K / D",
    "train.gamesPerSec": "Games / s",
    "train.aliveLanes": "Alive lanes",
    "train.bufferFill": "Replay buffer",
    "train.stats": "Training stats",
    "train.charts": "Training curves",
    "train.chanceLevel": "chance line",

    // Train tab — config form (ranges validated client-side, rebuilt worker-side)
    "train.config.title": "Network & training config",
    "train.config.arch": "Architecture",
    "train.config.arch.mlp": "MLP (dense stack)",
    "train.config.arch.attention": "Entity attention encoder",
    "train.config.hidden": "Hidden layers (comma-separated)",
    "train.config.lr": "Learning rate",
    "train.config.batch": "Batch size",
    "train.config.buffer": "Replay buffer (rows)",
    "train.config.gamma": "Discount γ",
    "train.config.epsDecay": "ε decay / episode",
    "train.config.epsMin": "Min ε",
    "train.config.lanes": "Self-play lanes",
    "train.config.trainEvery": "Train every N decisions",
    "train.config.targetSync": "Target sync every N steps",
    "train.config.evalEvery": "Evaluate every N episodes",
    "train.config.evalEpisodes": "Eval episodes",
    "train.config.episodeSeconds": "Episode length (s)",
    "train.config.tankCount": "Tanks per arena",
    "train.config.doubleQ": "Double DQN (DDQN)",
    "train.config.apply": "Apply (resets training)",
    "train.config.note": "Applying rebuilds the network and resets all training progress.",
    "train.config.invalid": "Invalid config — check the highlighted values.",

    // Play tab
    "play.opponents": "Opponents",
    "play.teamMode": "Team mode",
    "play.teamMode.ffa": "Free-for-all",
    "play.teamMode.allied": "Allied (AIs on your team)",
    "play.difficulty": "Difficulty",
    "play.aimAssist": "Aim assist",
    "play.rematch": "Rematch",
    "play.hint": "W/S throttle · A/D steer · Space fire",
    "play.untrained": "Untrained model (<10 episodes)",
    "play.error": "Could not start the match — please retry.",
    "play.banner.win": "Victory!",
    "play.banner.lose": "Defeated.",
    "play.banner.draw": "Mutual destruction — draw.",
    "play.banner.reconfigured": "Engine reconfigured — hit Rematch for a new match.",

    // Watch tab
    "watch.tanks": "Tanks",
    "watch.speed": "Speed",
    "watch.pause": "Pause",
    "watch.resume": "Resume",
    "watch.step": "Step",
    "watch.autoRestart": "Auto-restart",
    "watch.scoreboard": "Scoreboard",
    "watch.gen": "Model gen",
    "watch.error": "Could not start the spectator lane — please retry.",
    // {name} is a seat name (name.p1..p4).
    "watch.banner.winner": "{name} wins!",
    "watch.banner.draw": "Mutual destruction — draw.",

    // Escalation banner (phase 1 = powerups speed up; phase 2 = HP drain)
    "esc.phase1": "Escalation",
    "esc.phase2": "Sudden death",

    // Power-ups (order matches game/types.ts PowerUpKind)
    "powerup.rapid": "Rapid fire",
    "powerup.shield": "Shield",
    "powerup.speed": "Speed boost",
    "powerup.repair": "Repair",
    "powerup.bounce": "Bounce ammo",
    "powerup.pierce": "Pierce rounds",

    // HUD chips
    "hud.hp": "HP",
    "hud.reload": "Reload",
    "hud.kills": "Kills",
    "hud.deaths": "Deaths",

    // Seat names (watch scoreboard + play HUD)
    "name.p1": "P1",
    "name.p2": "P2",
    "name.p3": "P3",
    "name.p4": "P4",
  },
  zh: {
    "app.title": "戰車大戰 · TankWar",
    "app.subtitle": "自我對弈強化學習，即時在瀏覽器運行",
    "tab.train": "訓練",
    "tab.play": "對戰",
    "tab.watch": "觀戰",
    "status.loading": "引擎載入中…",
    "status.ready": "引擎就緒",
    "status.running": "訓練中",
    "header.language": "語言",
    "header.theme": "主題",
    "header.sound": "音效開",
    "header.soundOff": "音效關",

    "train.placeholder": "訓練控制與即時圖表將於學習階段加入——目前為骨架版本。",
    "play.placeholder": "對戰模式將於訓練出模型後加入——目前為骨架版本。",
    "watch.placeholder": "觀戰模式將於渲染器完成後加入——目前為骨架版本。",

    // 訓練分頁——開始／停止與數據格
    "train.start": "開始訓練",
    "train.stop": "停止訓練",
    "train.backend": "後端",
    "train.episodes": "回合數",
    "train.decisions": "決策數",
    "train.trainSteps": "訓練步數",
    "train.epsilon": "ε（探索率）",
    "train.loss": "損失（平滑）",
    "train.avgReturn": "平均回報",
    "train.winRate": "對隨機勝率",
    // {chance} 為每場戰車數 N——N 人混戰的合理基準就是 1/N。
    "train.winRateChance": "隨機基準 = 1/{chance}",
    "train.kd": "擊殺／死亡",
    "train.gamesPerSec": "每秒回合",
    "train.aliveLanes": "運行中賽道",
    "train.bufferFill": "回放緩衝區",
    "train.stats": "訓練數據",
    "train.charts": "訓練曲線",
    "train.chanceLevel": "隨機基準線",

    // 訓練分頁——設定表單（前端驗證範圍，重建在 worker 端進行）
    "train.config.title": "網絡與訓練設定",
    "train.config.arch": "網絡架構",
    "train.config.arch.mlp": "MLP（全連接堆疊）",
    "train.config.arch.attention": "實體注意力編碼器",
    "train.config.hidden": "隱藏層（逗號分隔）",
    "train.config.lr": "學習率",
    "train.config.batch": "批次大小",
    "train.config.buffer": "回放緩衝區（行數）",
    "train.config.gamma": "折扣因子 γ",
    "train.config.epsDecay": "ε 衰減（每回合）",
    "train.config.epsMin": "最小 ε",
    "train.config.lanes": "並行賽道",
    "train.config.trainEvery": "每 N 次決策訓練一次",
    "train.config.targetSync": "每 N 步同步目標網絡",
    "train.config.evalEvery": "每 N 回合評估",
    "train.config.evalEpisodes": "評估回合數",
    "train.config.episodeSeconds": "回合時長（秒）",
    "train.config.tankCount": "每場戰車數",
    "train.config.doubleQ": "雙 DQN（DDQN）",
    "train.config.apply": "套用（會重設訓練）",
    "train.config.note": "套用後會重建網絡並重設所有訓練進度。",
    "train.config.invalid": "設定無效——請檢查標示的數值。",

    // 對戰分頁
    "play.opponents": "對手數量",
    "play.teamMode": "隊伍模式",
    "play.teamMode.ffa": "自由混戰（FFA）",
    "play.teamMode.allied": "盟友模式（AI 同你一隊）",
    "play.difficulty": "難度",
    "play.aimAssist": "輔助瞄準",
    "play.rematch": "再來一場",
    "play.hint": "W/S 油門 · A/D 轉向 · 空白鍵開火",
    "play.untrained": "模型未經訓練（少於 10 回合）",
    "play.error": "無法開始對戰——請重試。",
    "play.banner.win": "勝利！",
    "play.banner.lose": "戰敗。",
    "play.banner.draw": "同歸於盡——平手。",
    "play.banner.reconfigured": "引擎已重新設定——按「再來一場」開始新對戰。",

    // 觀戰分頁
    "watch.tanks": "戰車數量",
    "watch.speed": "播放速度",
    "watch.pause": "暫停",
    "watch.resume": "繼續",
    "watch.step": "單步",
    "watch.autoRestart": "自動重開",
    "watch.scoreboard": "記分板",
    "watch.gen": "模型世代",
    "watch.error": "無法開始觀戰——請重試。",
    // {name} 為座位名稱（name.p1..p4）。
    "watch.banner.winner": "{name} 獲勝！",
    "watch.banner.draw": "同歸於盡——平手。",

    // 升級橫額（第一階段：強化道具加速出現；第二階段：突死模式）
    "esc.phase1": "戰況升級",
    "esc.phase2": "突死模式",

    // 強化道具（順序對應 game/types.ts 的 PowerUpKind）
    "powerup.rapid": "速射",
    "powerup.shield": "護盾",
    "powerup.speed": "加速",
    "powerup.repair": "維修",
    "powerup.bounce": "彈射彈藥",
    "powerup.pierce": "貫穿彈",

    // HUD 資訊格
    "hud.hp": "生命值",
    "hud.reload": "裝填",
    "hud.kills": "擊殺",
    "hud.deaths": "死亡",

    // 座位名稱（觀戰記分板與對戰 HUD）
    "name.p1": "玩家一",
    "name.p2": "玩家二",
    "name.p3": "玩家三",
    "name.p4": "玩家四",
  },
} as const;

// The `as const` EN dict defines the key set; the assignment below forces the
// zh dict to repeat every key (and only those keys) — a missing or misspelled
// translation is a compile error, not a blank label at runtime.
export type StringKey = keyof typeof STRINGS.en;
const zhParity: Record<StringKey, string> = STRINGS.zh;
void zhParity;

/** Translate `key` in `lang`, interpolating `{param}` placeholders. */
export function t(
  lang: Lang,
  key: StringKey,
  params?: Record<string, string | number>,
): string {
  const dict = STRINGS[lang] as Record<StringKey, string>;
  let s = dict[key] ?? (STRINGS.en as Record<StringKey, string>)[key] ?? key;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      s = s.replaceAll(`{${k}}`, String(v));
    }
  }
  return s;
}

/** Browser language → default UI language (zh for any Chinese locale). */
export function detectLang(): Lang {
  if (typeof navigator === "undefined") return "en";
  return navigator.language?.toLowerCase().startsWith("zh") ? "zh" : "en";
}

/** Persisted language choice, falling back to browser detection. */
export function loadLang(): Lang {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "en" || v === "zh") return v;
  } catch {
    // localStorage unavailable (private mode etc.) — fall through.
  }
  return detectLang();
}

export function saveLang(lang: Lang): void {
  try {
    localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    // Best-effort persistence only.
  }
}
