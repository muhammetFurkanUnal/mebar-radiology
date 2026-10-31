from pathlib import Path
import shutil
import torch

def prep_run_folder(cwd, config):
    cwd = Path(cwd)
    config_path = cwd / "config.json"
    run_dir = cwd / "runs" / config.name
    (run_dir / "ckps").mkdir(parents=True, exist_ok=True)

    dst = run_dir / "config.json"
    if not dst.exists():
        shutil.copy2(config_path, dst)

    return run_dir

class CheckpointManager:
    """Model-agnostic checkpoint manager.
 
    Works with anything that exposes state_dict() / load_state_dict():
    models, optimizers, schedulers.
    Checkpoints go to <script_dir>/runs/<run_name>/ckps/ep<N>.pt
    """
    def __init__(self, script_dir, run_name, model, optimizer=None, scheduler=None):
        script_dir = Path(script_dir)
        if not script_dir.is_dir():
            raise FileNotFoundError(f"Script folder not found: {script_dir}")
 
        self.ckpt_dir = script_dir / "runs" / run_name / "ckps"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
 
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
 
    def save(self, epoch, best=False):
        state = {"epoch": epoch, "model": self.model.state_dict()}
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()

        path = self.ckpt_dir / f"ep{epoch + 1}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

        if best:
            shutil.copy2(path, self.ckpt_dir / "best.pt")
 
    def latest_epoch(self):
        """Highest N among saved ep<N>.pt files, or None if there are none."""
        epochs = [int(p.stem[2:]) for p in self.ckpt_dir.glob("ep*.pt")
                  if p.stem[2:].isdigit()]
        return max(epochs) if epochs else None
 
    def load(self, which="latest", model_only=False, strict=True):
        """which: an epoch number (e.g. 50 -> ep50.pt) or "latest".
        With model_only=True, optimizer and scheduler states are not
        restored (for evaluation / inference).
        Returns the saved (0-indexed) epoch."""
        if which == "latest":
            n = self.latest_epoch()
            if n is None:
                raise FileNotFoundError(f"No checkpoints in {self.ckpt_dir}")
            name = f"ep{n}"
        elif which == "best":
            name = "best"
        else:
            name = f"ep{int(which)}"
        
        path = self.ckpt_dir / f"{name}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
 
        self.model.load_state_dict(state["model"], strict=strict)
        if not model_only:
            if self.optimizer is not None and "optimizer" in state:
                self.optimizer.load_state_dict(state["optimizer"])
            if self.scheduler is not None and "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
 
        return state["epoch"]