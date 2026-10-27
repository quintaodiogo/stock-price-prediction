"""Rede sequencial (GRU) no walk-forward: etapa 3 da Pessoa B.

Diferença para o LightGBM: em vez de ver só as features de hoje, a rede vê uma janela com as
features dos últimos `L` pregões do ativo (L = 40 por padrão) e prevê o alvo de hoje.

- Um modelo geral por fold e alvo, com todos os ativos juntos (mesmo papel do `lgbm_geral`).
- Features: as mesmas `f_*` do painel, padronizadas com média e desvio **só do treino** do fold;
  ausentes (f_range, f_volrel) viram 0 depois da padronização.
- Validação interna igual à do `tree.py` (últimos 20% das datas do treino, com embargo);
  early stopping pela perda na validação, e fica o peso da melhor época.
- Começa pela volatilidade (`y_vol_h5`), o alvo com sinal no experimento 1b.

Uso:
  python src/models/seq.py                                 (y_vol_h5, todos os folds)
  python src/models/seq.py --targets y_vol_h5 y_ret_h1
  python src/models/seq.py --exclude SP500 NASDAQ DOWJONES  (cenário "ativo não visto")
  python src/models/seq.py --folds 0 1 --epochs 3          (teste rápido)
-> reports/results/preds_gru_geral[_sem_<ativos>].parquet  (pesos em models/)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from tree import inner_split, tag  # noqa: E402

CONFIG = {
    "window": 40,          # pregões por janela
    "hidden": 32,
    "layers": 1,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch": 512,
    "max_epochs": 30,
    "patience": 4,
}


# ---------------------------------------------------------------- janelas -----------------
class Windows:
    """Guarda as features de todos os ativos numa matriz só, ordenada por (ativo, data).

    A janela que termina na linha i é formada pelas linhas i-L+1 .. i, que precisam ser do mesmo
    ativo. Nenhuma linha posterior a i entra, então a janela só usa dados até a data da previsão.
    """

    def __init__(self, panel: pd.DataFrame, feats: list[str], window: int):
        self.panel = panel.sort_values(["asset", "date"]).reset_index(drop=True)
        self.feats, self.window = feats, window
        asset = self.panel["asset"].to_numpy()
        pos = self.panel.groupby("asset").cumcount().to_numpy()
        self.valid = pos >= window - 1                       # há histórico suficiente
        self.offsets = np.arange(window - 1, -1, -1)         # L-1, ..., 0
        self.raw = np.array(self.panel[feats], dtype=np.float32)   # cópia gravável (pandas 3 devolve só leitura)
        assert (asset[np.where(self.valid)[0][:, None] - self.offsets] ==
                asset[np.where(self.valid)[0]][:, None]).all(), "janela cruzou ativos"

    def standardize(self, train_rows: np.ndarray) -> torch.Tensor:
        """Padroniza com média/desvio das linhas de treino; ausente vira 0."""
        mu = np.nanmean(self.raw[train_rows], axis=0)
        sd = np.nanstd(self.raw[train_rows], axis=0)
        sd[~(sd > 0)] = 1.0
        x = np.nan_to_num((self.raw - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x.astype(np.float32))

    def batch(self, x: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
        """(B, L, F): para cada linha final, as L linhas até ela."""
        idx = torch.from_numpy(rows[:, None] - self.offsets)
        return x[idx]


# ---------------------------------------------------------------- modelo ------------------
class GRUNet(nn.Module):
    def __init__(self, n_feat: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def loss_fn(target: str) -> nn.Module:
    return nn.BCEWithLogitsLoss() if target == "y_dir_h1" else nn.MSELoss()


def predict(model: nn.Module, win: Windows, x: torch.Tensor, rows: np.ndarray, target: str) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 4096):
            out.append(model(win.batch(x, rows[i:i + 4096])))
    pred = torch.cat(out) if out else torch.empty(0)
    return (torch.sigmoid(pred) if target == "y_dir_h1" else pred).numpy()


def train_one(win: Windows, x: torch.Tensor, y: np.ndarray, fit_rows: np.ndarray, val_rows: np.ndarray,
              target: str, cfg: dict) -> tuple[nn.Module, dict]:
    torch.manual_seed(common.SEED)
    rng = np.random.default_rng(common.SEED)
    model = GRUNet(x.shape[1], cfg["hidden"], cfg["layers"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    crit = loss_fn(target)
    y_t = torch.from_numpy(y.astype(np.float32))
    best, best_state, bad, hist = np.inf, None, 0, []
    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = rng.permutation(fit_rows)                     # ordem aleatória a cada época
        for i in range(0, len(perm), cfg["batch"]):
            rows = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            loss = crit(model(win.batch(x, rows)), y_t[rows])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([crit(model(win.batch(x, val_rows[j:j + 4096])), y_t[val_rows[j:j + 4096]]).item()
                                for j in range(0, len(val_rows), 4096)]))
        hist.append(round(vl, 5))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, {"epocas": len(hist), "melhor_epoca": int(np.argmin(hist)) + 1, "val_loss": hist}


# ---------------------------------------------------------------- walk-forward ------------
def run(panel: pd.DataFrame, folds: list[dict], targets: list[str], exclude: list[str], cfg: dict,
        save_models: bool, name: str) -> tuple[list[pd.DataFrame], list[dict]]:
    feats = common.feature_cols(panel)
    win = Windows(panel, feats, cfg["window"])
    p = win.panel
    dates, assets = p["date"].to_numpy(), p["asset"].to_numpy()
    frames, log = [], []
    for fold in folds:
        t0 = time.time()
        tr_end = np.datetime64(pd.Timestamp(fold["train_end"]))
        te_lo, te_hi = (np.datetime64(pd.Timestamp(fold[k])) for k in ("test_start", "test_end"))
        in_train = (dates <= tr_end) & ~np.isin(assets, exclude)
        x = win.standardize(np.where(in_train)[0])
        test_rows = np.where(win.valid & (dates >= te_lo) & (dates <= te_hi))[0]
        for target in targets:
            y = np.array(p[target], dtype=np.float32)
            ok = win.valid & in_train & ~np.isnan(y)
            fit_df, val_df = inner_split(p[ok])
            fit_rows, val_rows = np.array(fit_df.index), np.array(val_df.index)
            if len(fit_rows) < 1000 or len(val_rows) == 0 or len(test_rows) == 0:
                continue
            model, info = train_one(win, x, y, fit_rows, val_rows, target, cfg)
            pred = predict(model, win, x, test_rows, target)
            frames.append(common.preds_frame(p.iloc[test_rows], fold, target, pred))
            log.append({"fold": fold["fold"], "target": target, **info})
            if save_models:
                common.MODELS_DIR.mkdir(exist_ok=True)
                torch.save(model.state_dict(), common.MODELS_DIR / f"{name}_{target}_fold{fold['fold']}.pt")
        print(f"  fold {fold['fold']:2d} ({fold['test_start'][:4]}) ok  {time.time() - t0:.0f}s", flush=True)
    return frames, log


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["y_vol_h5"], choices=common.TARGETS)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="só estes folds (teste rápido)")
    ap.add_argument("--epochs", type=int, default=None, help="máximo de épocas (padrão 30)")
    ap.add_argument("--window", type=int, default=None, help="pregões por janela (padrão 40)")
    ap.add_argument("--no-save-models", action="store_true")
    args = ap.parse_args(argv)

    cfg = {**CONFIG, **{k: v for k, v in (("max_epochs", args.epochs), ("window", args.window)) if v}}
    torch.set_num_threads(max(1, torch.get_num_threads()))
    panel, folds = common.load_panel(), common.load_folds()
    if args.folds is not None:
        folds = [f for f in folds if f["fold"] in args.folds]
    unknown = set(args.exclude) - set(panel["asset"])
    if unknown:
        sys.exit(f"Ativos fora do painel: {sorted(unknown)}")
    name = "gru_" + tag("geral", args.exclude)
    print(f"{name}: {len(folds)} folds, alvos {args.targets}, janela {cfg['window']}")
    t0 = time.time()
    frames, log = run(panel, folds, args.targets, args.exclude, cfg, not args.no_save_models, name)
    info = {"modelo": name, "config": cfg, "excluidos": args.exclude, "treino": log,
            "painel": common.panel_signature(panel, common.load_folds())}
    path = common.save_preds(frames, name, info)
    print(f"\n=== {name}: {time.time() - t0:.0f}s -> {path.relative_to(common.ROOT)}")
    print(common.quick_metrics(pd.read_parquet(path)).to_string())


if __name__ == "__main__":
    main()
