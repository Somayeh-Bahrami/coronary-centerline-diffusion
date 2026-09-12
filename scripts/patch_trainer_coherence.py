"""Apply the edge-coherence wiring to src/coronarycl/trainer.py.

Idempotent and fully asserted: every anchor must appear exactly once or the
script refuses to write anything. Nothing else changes -- diffusion.py,
sampling.py and dataset_v3_1.py are untouched. The edge lookup keys off
batch["sample"] in a collate object, so the dataset class needs no edits.

DESIGN NOTE THAT MATTERS FOR THE EXPERIMENT
-------------------------------------------
The coherence term is added ONLY when fixed_t is None. evaluate() always passes
fixed_t, so the deterministic VAL loss keeps its exact previous definition and
stays comparable with every earlier run and with the control arm. The auxiliary
term must never leak into the number used for checkpointing or reporting.

Run from the repo root:  python scripts/patch_trainer_coherence.py
"""
import ast
import sys
from pathlib import Path

TRAINER = Path("src/coronarycl/trainer.py")

EDITS = [
    # 1. imports
    ("from .models.diffusion import CenterlineDenoiser\n",
     "from .models.diffusion import CenterlineDenoiser\n"
     "from .edge_coherence import (EdgeCollate, edge_coherence_loss,\n"
     "                             load_edge_cache, x0_from_eps)\n"),

    # 2. compute_loss signature
    ("def compute_loss(\n"
     "    model, scheduler, batch, device, fixed_t=None,\n"
     "    cond_drop_prob=0.0, self_cond_p=0.5, return_parts=False,\n"
     "):",
     "def compute_loss(\n"
     "    model, scheduler, batch, device, fixed_t=None,\n"
     "    cond_drop_prob=0.0, self_cond_p=0.5, return_parts=False,\n"
     "    coh_weight=0.0, coh_kwargs=None, return_coh=False,\n"
     "    detach_parts=True,\n"
     "):"),

    # 3. compute_loss tail
    ("    numerator = (per_point * mask.float()).sum()\n"
     "    denominator = mask.sum(dtype=torch.float32).clamp_min(1.0)\n"
     "    if return_parts:\n"
     "        return numerator, denominator\n"
     "    return numerator / denominator\n",
     "    numerator = (per_point * mask.float()).sum()\n"
     "    denominator = mask.sum(dtype=torch.float32).clamp_min(1.0)\n"
     "    if return_parts:\n"
     "        return numerator, denominator\n"
     "    epsilon_loss = numerator / denominator\n"
     "\n"
     "    # Coherence is a TRAINING-only term. evaluate() always passes fixed_t,\n"
     "    # so the deterministic VAL loss keeps its previous definition and stays\n"
     "    # comparable with earlier runs and with the no-coherence control arm.\n"
     "    coherence_loss = torch.zeros((), device=device)\n"
     "    if coh_weight > 0.0 and fixed_t is None:\n"
     "        if batch.get('edge_index') is None:\n"
     "            # Never degrade silently to the epsilon-only objective: that\n"
     "            # would produce a run that LOOKS like the coherence arm and is\n"
     "            # actually the control.\n"
     "            raise RuntimeError(\n"
     "                'coh_weight > 0 but this batch carries no edge_index. The '\n"
     "                'train loader is missing its EdgeCollate collate_fn, or the '\n"
     "                'edge cache does not cover these samples.')\n"
     "        alpha_bar_t = scheduler.alpha_bars[timesteps]\n"
     "        x0_hat = x0_from_eps(noisy, predicted_noise, alpha_bar_t)\n"
     "        coherence_loss = edge_coherence_loss(\n"
     "            x0_hat, x0,\n"
     "            batch['edge_index'].to(device, non_blocking=True),\n"
     "            batch['edge_mask'].to(device, non_blocking=True),\n"
     "            alpha_bar_t, **(coh_kwargs or {}))\n"
     "        total_loss = epsilon_loss + coh_weight * coherence_loss\n"
     "    else:\n"
     "        total_loss = epsilon_loss\n"
     "\n"
     "    if return_coh:\n"
     "        if detach_parts:\n"
     "            return total_loss, epsilon_loss.detach(), coherence_loss.detach()\n"
     "        # graph-attached parts, for gradient_norm_ratio() only\n"
     "        return total_loss, epsilon_loss, coherence_loss\n"
     "    return total_loss\n"),

    # 4. config reads
    ('    checkpoint_dir = Path(train_cfg.get(\n'
     '        "checkpoint_dir", "outputs/h384_200k/checkpoints"))\n',
     '    checkpoint_dir = Path(train_cfg.get(\n'
     '        "checkpoint_dir", "outputs/h384_200k/checkpoints"))\n'
     '    coh_weight = float(train_cfg.get("coh_weight", 0.0))\n'
     '    edge_cache_path = train_cfg.get("edge_cache")\n'
     '    coh_kwargs = {\n'
     '        "huber_beta": float(train_cfg.get("coh_huber_beta", 0.05)),\n'
     '        "weight_by_abar": bool(train_cfg.get("coh_weight_by_abar", True)),\n'
     '        "hinge_k": float(train_cfg.get("coh_hinge_k", 0.0)),\n'
     '        "hinge_weight": float(train_cfg.get("coh_hinge_weight", 0.0)),\n'
     '    }\n'
     '    if coh_weight > 0.0 and not edge_cache_path:\n'
     '        raise ValueError("train.coh_weight > 0 requires train.edge_cache")\n'),

    # 5. build the collate and record the setting in the run signature
    ('    loader_kwargs = {\n',
     '    edge_collate = None\n'
     '    if coh_weight > 0.0:\n'
     '        edge_map, edge_meta = load_edge_cache(\n'
     '            edge_cache_path, packaged_dir,\n'
     '            required_ids=list(train_ids))\n'
     '        edge_collate = EdgeCollate(edge_map)\n'
     '        print(f"edge cache: {edge_meta[\'n_samples\']} samples, "\n'
     '              f"fingerprints verified, coh_weight={coh_weight:g}, "\n'
     '              f"{coh_kwargs}")\n'
     '\n'
     '    loader_kwargs = {\n'),

    # VAL deliberately gets NO edge collate: the deterministic VAL loss is
    # epsilon-only, so attaching edges there would make validation depend on a
    # cache it does not need and crash whenever the cache omits val samples.
    ('    train_loader = DataLoader(\n'
     '        train_dataset,\n'
     '        batch_sampler=DeterministicStepBatchSampler(\n'
     '            len(train_dataset), batch_size, step, max_steps, seed + 10_000),\n'
     '        generator=torch.Generator().manual_seed(seed + 2_000_000),\n'
     '        **loader_kwargs)',
     '    train_loader_kwargs = dict(loader_kwargs)\n'
     '    if edge_collate is not None:\n'
     '        train_loader_kwargs["collate_fn"] = edge_collate\n'
     '    train_loader = DataLoader(\n'
     '        train_dataset,\n'
     '        batch_sampler=DeterministicStepBatchSampler(\n'
     '            len(train_dataset), batch_size, step, max_steps, seed + 10_000),\n'
     '        generator=torch.Generator().manual_seed(seed + 2_000_000),\n'
     '        **train_loader_kwargs)'),

    ('        "dataset_fingerprint": _dataset_fingerprint(\n'
     '            packaged_dir, train_ids, val_ids),\n'
     '    }',
     '        "dataset_fingerprint": _dataset_fingerprint(\n'
     '            packaged_dir, train_ids, val_ids),\n'
     '        "coh_weight": coh_weight,\n'
     '        "coh_kwargs": coh_kwargs,\n'
     '    }'),

    # 6. history keys
    ('    history = {"train_step": [], "train_loss": [], "val_step": [],\n'
     '               "val_loss": [], "lr": []}',
     '    history = {"train_step": [], "train_loss": [], "val_step": [],\n'
     '               "val_loss": [], "lr": [], "eps_loss": [], "coh_loss": []}'),

    # 7. training step: capture both parts
    ('        with _autocast_context(device, precision):\n'
     '            loss = compute_loss(\n'
     '                model, noise_scheduler, batch, device,\n'
     '                cond_drop_prob=cond_drop_prob, self_cond_p=self_cond_p)\n',
     '        with _autocast_context(device, precision):\n'
     '            loss, eps_part, coh_part = compute_loss(\n'
     '                model, noise_scheduler, batch, device,\n'
     '                cond_drop_prob=cond_drop_prob, self_cond_p=self_cond_p,\n'
     '                coh_weight=coh_weight, coh_kwargs=coh_kwargs,\n'
     '                return_coh=True)\n'),

    # 8. log both parts
    ('            history["train_step"].append(step)\n'
     '            history["train_loss"].append(loss_value)\n',
     '            history["train_step"].append(step)\n'
     '            history["train_loss"].append(loss_value)\n'
     '            history["eps_loss"].append(float(eps_part))\n'
     '            history["coh_loss"].append(float(coh_part))\n'),

    # 9. print both parts at every validation
    ('            print(\n'
     '                f"step {step}: train_loss={current_train_loss:.6f}, "\n'
     '                f"val_loss={last_val_loss:.6f}, "\n'
     '                f"lr={optimizer.param_groups[0][\'lr\']:.3e}, "\n'
     '                f"session_elapsed={time.time() - session_started:.1f}s")\n',
     '            parts = ""\n'
     '            if coh_weight > 0.0:\n'
     '                parts = (f", eps={float(eps_part):.6f}"\n'
     '                         f", coh={float(coh_part):.6f}"\n'
     '                         f", w*coh={coh_weight * float(coh_part):.6f}")\n'
     '            print(\n'
     '                f"step {step}: train_loss={current_train_loss:.6f}{parts}, "\n'
     '                f"val_loss={last_val_loss:.6f}, "\n'
     '                f"lr={optimizer.param_groups[0][\'lr\']:.3e}, "\n'
     '                f"session_elapsed={time.time() - session_started:.1f}s")\n'),
]


def main():
    if not TRAINER.exists():
        sys.exit(f"run from the repo root; {TRAINER} not found")
    src = TRAINER.read_text()
    if "edge_coherence_loss" in src:
        print("already patched; nothing to do")
        return
    for old, new in EDITS:
        n = src.count(old)
        if n != 1:
            sys.exit(f"REFUSED: anchor appears {n} times, expected 1:\n{old[:160]}")
        src = src.replace(old, new)
    ast.parse(src)
    TRAINER.write_text(src)
    print(f"patched {TRAINER} ({len(EDITS)} edits), parses OK")
    print("coh_weight defaults to 0.0 -> behaviour is bit-identical until you set it")


if __name__ == "__main__":
    main()
