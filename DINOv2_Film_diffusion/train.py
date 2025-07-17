import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import glob
from contextlib import nullcontext
import matplotlib.pyplot as plt
from .diffusion import ConditionalDiffusionModel

class TensorImageDataset(Dataset):
    def __init__(
        self,
        data_path,
        image_size=224,
        max_tensor_len=128,
        feature_dim=3
    ):
        
        self.data_path = data_path
        self.max_tensor_len = max_tensor_len
        self.feature_dim = feature_dim
        self.image_size = image_size
        
        self.sample_pairs = []
        
        for folder_a in sorted(glob.glob(os.path.join(data_path, "*"))):
            if not os.path.isdir(folder_a):
                continue
                
            for folder_b in sorted(glob.glob(os.path.join(folder_a, "*"))):
                if not os.path.isdir(folder_b):
                    continue
                    
                npy_files = glob.glob(os.path.join(folder_b, "*.npy"))
                npz_files = glob.glob(os.path.join(folder_b, "*.npz"))
                
                if len(npy_files) >= 1 and len(npz_files) >= 1:
                    self.sample_pairs.append({
                        'folder': folder_b,
                        'npy': npy_files[0],
                        'npz': npz_files[0]
                    })
        
        print(f"loaded {len(self.sample_pairs)} samples")
        
    def __len__(self):
        return len(self.sample_pairs)
    
    def __getitem__(self, idx):
        sample = self.sample_pairs[idx]
        img_path = sample['npy']
        tensor_path = sample['npz']
        
        # Loading image data
        image = np.load(img_path)

        if image.ndim == 3 and image.shape[0] not in [1, 3]:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 2:  
            image = np.expand_dims(image, 0)

        image = image.astype(np.float32)

        image = image / 255.0

        image = (image - 0.5) / 0.5

        image = torch.from_numpy(image).float()
        
        # Adjust image size
        if image.dim() == 3:
            _, h, w = image.shape
            if h != self.image_size or w != self.image_size:
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), 
                    size=(self.image_size, self.image_size), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
   
        # loading tensor data
        try:
            npz_file = np.load(tensor_path)

            if isinstance(npz_file, np.lib.npyio.NpzFile):
                if 'data' in npz_file.files: 
                    tensor_data_raw = npz_file['data']
                else: 
                    tensor_data_raw = npz_file[npz_file.files[0]]
                
                if tensor_data_raw.dtype.type is np.str_ or tensor_data_raw.dtype.type is np.string_:
                    tensor_data = []
                    for line in tensor_data_raw:
                        values = line.strip().split()
                        if len(values) >= 3:
                            tensor_data.append([float(values[0]), float(values[1]), float(values[2])])
                    tensor_data = np.array(tensor_data)
                else:
                    tensor_data = tensor_data_raw
            else:
                tensor_data = npz_file
                
        except Exception as e:
            print(f"Cannot load as npz{tensor_path},try to load as txt: {str(e)}")
            tensor_data = []
            try:
                with open(tensor_path, 'r') as f:
                    for line in f:
                        values = line.strip().split()
                        if len(values) >= 3:
                            tensor_data.append([float(values[0]), float(values[1]), float(values[2])])
                tensor_data = np.array(tensor_data)
            except Exception as e2:
                print(f"txt loading fail: {str(e2)}")
                tensor_data = np.zeros((0, self.feature_dim))

        if tensor_data.shape[0] > self.max_tensor_len:
            tensor_data = tensor_data[:self.max_tensor_len]
        
        original_length = tensor_data.shape[0]
        
        # padding tensor data
        padded_tensor = np.zeros((self.max_tensor_len, self.feature_dim))
        padded_tensor[:original_length] = tensor_data
        
        # construct masl
        mask = np.zeros(self.max_tensor_len)
        mask[:original_length] = 1.0
        
    
        tensor_data_transposed = np.transpose(padded_tensor, (1, 0))
        
        return {
            'image': image,  # [C, H, W]
            'tensor': torch.from_numpy(tensor_data_transposed).float(),  # [C, L]
            'mask': torch.from_numpy(mask).float(),
            'length': original_length,
            'path': sample['folder']
        }


FIXED_BATCH = None


def visualize_predictions(model, data_loader, device,
                          epoch: int, save_dir: str,
                          debug: bool = False):

    import os, numpy as np, torch, matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    global FIXED_BATCH
    if epoch % 5 != 0:      # 只在 5 的倍数 epoch 可视化
        return

    viz_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # ── 取一批固定样本 ───────────────────────────────────────────
    num_samples = min(10, len(data_loader.dataset))
    if FIXED_BATCH is None:
        tmp_loader = DataLoader(
            data_loader.dataset, batch_size=num_samples, shuffle=True,
            num_workers=data_loader.num_workers,
            pin_memory=getattr(data_loader, 'pin_memory', False))
        FIXED_BATCH = next(iter(tmp_loader))
    batch = FIXED_BATCH

    images = batch["image"].to(device)            # [B,C,H,W]
    true_tensors_t = batch["tensor"].cpu()        # [B,C,L]
    true_np = torch.transpose(true_tensors_t, 1, 2).numpy()  # [B,L,C]

    # ── 预测 ───────────────────────────────────────────────────
    with torch.no_grad():
        pred_tensors = model.sample(batch_size=num_samples,
                                    cond_img=images)
    pred_np = pred_tensors.cpu().transpose(1, 2).numpy()

    # ── 先算整批平均 MSE ───────────────────────────────────────
    mse_xy_list, mse_xyz_list = [], []
    for t, p in zip(true_np, pred_np):
        L = min(t.shape[0], p.shape[0])           # 长度对齐
        mse_xy_list.append(np.mean((t[:L, :2] - p[:L, :2]) ** 2))
        mse_xyz_list.append(np.mean((t[:L, :3] - p[:L, :3]) ** 2))

    avg_mse_xy  = float(np.mean(mse_xy_list))
    avg_mse_xyz = float(np.mean(mse_xyz_list))

    # ── 逐样本绘图 ─────────────────────────────────────────────
    for i in range(num_samples):
        true = true_np[i]
        pred = pred_np[i]
        L = min(true.shape[0], pred.shape[0])
        mse_xy  = mse_xy_list[i]
        mse_xyz = mse_xyz_list[i]

        fig = plt.figure(figsize=(12, 10))
        fig.suptitle(
            f"Sample MSE(XY)={mse_xy:.4e} | "
            f"MSE(XYE)={mse_xyz:.4e}\n"
            f"Batch Avg MSE(XY)={avg_mse_xy:.4e} | "
            f"Batch Avg MSE(XYE)={avg_mse_xyz:.4e}",
            fontsize=13)

        # (1) 输入图片
        ax = fig.add_subplot(2, 2, 1)
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        if img.min() < 0:           # 归一化回 [0,1]
            img = (img + 1) / 2
        ax.imshow(img); ax.axis("off"); ax.set_title("Input Image")

        # (2) 真值张量
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(true[:L, 0], true[:L, 1],
                    c=true[:L, 2], cmap="viridis", s=20)
        ax2.set_title("True Tensor"); ax2.set_xlim(-1,1); ax2.set_ylim(-1,1); ax2.set_aspect('equal')

        # (3) 预测张量
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(pred[:L, 0], pred[:L, 1],
                    c=pred[:L, 2], cmap="viridis", s=20)
        ax3.set_title("Predicted Tensor"); ax3.set_xlim(-1,1); ax3.set_ylim(-1,1); ax3.set_aspect('equal')

        # (4) 对比
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.scatter(true[:L, 0], true[:L, 1],
                    c="blue", s=20, alpha=0.5, label="True")
        ax4.scatter(pred[:L, 0], pred[:L, 1],
                    c="red",  s=20, alpha=0.5, label="Pred")
        ax4.set_title("Comparison"); ax4.legend()
        ax4.set_xlim(-1,1); ax4.set_ylim(-1,1); ax4.set_aspect('equal')

        fig.tight_layout(rect=[0,0.05,1,0.9])
        fname = os.path.join(viz_dir, f"epoch_{epoch}_sample_{i}.png")
        fig.savefig(fname, dpi=150); plt.close(fig)

    if debug:
        print(f"Visualizations saved to {viz_dir}")



def plot_training_history(history, save_dir):
    train_epochs = [x['epoch'] + x['step']/1000. for x in history['train_loss']]
    train_losses = [x['loss'] for x in history['train_loss']]
    val_epochs = [x['epoch'] for x in history['val_loss']]
    val_losses = [x['loss'] for x in history['val_loss']]

    # Loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(train_epochs, train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()


import argparse
import time
import wandb


def train_conditional_diffusion(
    model: ConditionalDiffusionModel,
    data_path: str,
    save_dir: str = "./checkpoints3",
    batch_size: int = 32,
    image_size: int = 224,
    max_tensor_len: int = 128,
    feature_dim: int = 3,
    num_workers: int = 20,
    epochs: int = 500,
    lr: float = 1e-5,
    device: str = "cuda",
    log_every: int = 10,
    save_every: int = 10,
    eval_ratio: float = 0.1,
    resume_path: str | None = None,
    resume_optimizer: bool = True,
    resume_scheduler: bool = True,
    resume_epoch: bool = True,
    seed: int = 42,
    use_wandb: bool = True,
    wandb_project: str = "conditional-diffusion",
    wandb_name: str = None,
):
    # 训练函数主体保持不变，因为 FiLM 替换是在模型定义部分完成的
    # 训练流程与原始的 CrossAttention 模型相同
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    
    # Variables to store wandb ID
    wandb_run_id = None
    
    # Check if resuming and load wandb ID if available
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        if "wandb_run_id" in ckpt and use_wandb:
            wandb_run_id = ckpt["wandb_run_id"]
    
    # Initialize wandb
    if use_wandb:
        run_name = wandb_name or f"diffusion-film-{time.strftime('%Y%m%d-%H%M%S')}"  # 添加 film 标识
        if wandb_run_id:
            # Resume existing wandb run
            wandb.init(project=wandb_project, name=run_name, id=wandb_run_id, resume="allow")
            print(f"Resuming wandb run with ID: {wandb_run_id}")
        else:
            # Start new wandb run
            wandb.init(project=wandb_project, name=run_name, config={
                "batch_size": batch_size,
                "image_size": image_size,
                "max_tensor_len": max_tensor_len,
                "feature_dim": feature_dim,
                "epochs": epochs,
                "lr": lr,
                "seed": seed,
                "model_type": "FiLM",  # 添加模型类型标识
            })
            wandb_run_id = wandb.run.id
            print(f"Started new wandb run with ID: {wandb_run_id}")

    # Load dataset
    dataset = TensorImageDataset(
        data_path=data_path,
        image_size=image_size,
        max_tensor_len=max_tensor_len,
        feature_dim=feature_dim,
    )
    eval_size = int(len(dataset) * eval_ratio)
    train_size = len(dataset) - eval_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        ) if eval_size else None
    )

    print(f"train_set: {train_size} | val_set: {eval_size}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=20,
        min_lr=1e-8
    )

    history = {"train_loss": [], "val_loss": []}
    start_epoch = 0

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    
        if resume_optimizer and "optimizer_state_dict" in ckpt:
            print(f"LR before loading: {[group['lr'] for group in optimizer.param_groups]}")
        
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            print(f"LR after loading: {[group['lr'] for group in optimizer.param_groups]}")
            print(f"optimizer includes {len(optimizer.state_dict()['state'])} parameters")
    
        if resume_scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    
        if resume_epoch and "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
    
        if "history" in ckpt:
            history = ckpt["history"]
    
        print(f"Resumed from {resume_path}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}", leave=False
        )
        for step, batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            images = batch["image"].to(device)
            tensors = batch["tensor"].to(device)
            masks = batch.get("mask", None)
            if masks is not None:
                masks = masks.to(device)

            loss = model(x_seq=tensors, cond_img=images, mask=masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{running_loss/(step+1):.4f}"})
            if step % log_every == 0:
                history["train_loss"].append({
                    "epoch": epoch,
                    "step": step,
                    "loss": loss.item()
                })
                
                # Log to wandb
                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/step": step + epoch * len(train_loader),
                        "train/epoch": epoch
                    })

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} ▶ train_loss={avg_train_loss:.4f}")
        
        # Log average train loss
        if use_wandb:
            wandb.log({
                "train/avg_loss": avg_train_loss,
                "train/epoch": epoch
            })

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    tensors = batch["tensor"].to(device)
                    masks = batch.get("mask", None)
                    if masks is not None:
                        masks = masks.to(device)
                    loss = model(x_seq=tensors, cond_img=images, mask=masks)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            history["val_loss"].append({
                "epoch": epoch,
                "loss": val_loss
            })
            print(f"Validation ▶ val_loss={val_loss:.4f}")
            
            # Log validation loss
            if use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/epoch": epoch,
                    "lr": optimizer.param_groups[0]['lr']  # Log current learning rate
                })

            visualize_predictions(
                model=model,
                data_loader=val_loader,
                device=device,
                epoch=epoch+1,
                save_dir=save_dir,
                debug=False
            )
        else:
            scheduler.step(avg_train_loss)

        if (epoch + 1) % save_every == 0:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "wandb_run_id": wandb_run_id if use_wandb else None,  # Save wandb run ID
            }
            torch.save(ckpt, os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))
            torch.save(ckpt, os.path.join(save_dir, "model_latest.pt"))
            print("Checkpoints saved.")

    # Close wandb
    if use_wandb:
        wandb.finish()
        
    plot_training_history(history, save_dir)
    return model, history


def main():
    parser = argparse.ArgumentParser(description="conditional DINOv2 diffusion model with FiLM")  # 更新描述
    parser.add_argument("--data_path", type=str, default="/home/ziyue/Desktop/DATASET/diffusion_dataset", help="dataset path")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_film", help="folder to save checkpoints")  # 更新默认保存路径
    parser.add_argument("--batch_size", type=int, default=384, help="batch size")
    parser.add_argument("--image_size", type=int, default=224, help="img size")
    parser.add_argument("--max_tensor_len", type=int, default=128, help="max tensor length")
    parser.add_argument("--feature_dim", type=int, default=3, help="tensor feature dim")
    parser.add_argument("--num_workers", type=int, default=20, help="num workers")
    parser.add_argument("--epochs", type=int, default=500, help="training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--resume", type=str, default="", help="resume from checkpoint")  # 设置为空，避免使用CrossAttention的checkpoint
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="validation set ratio")
    parser.add_argument("--log_every", type=int, default=10, help="log every N steps")
    parser.add_argument("--save_every", type=int, default=5, help="save checkpoint every N epochs")
    parser.add_argument("--unet_dim", type=int, default=16, help="UNet base dim")
    parser.add_argument("--unet_dim_mults", type=str, default="1,2,2,4,8,8,16", help="UNet dim multipliers")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--timesteps", type=int, default=500, help="diffusion timesteps")
    parser.add_argument("--sampling_timesteps", type=int, default=500, help="sampling timesteps")
    parser.add_argument("--objective", type=str, default="pred_x0", choices=["pred_noise", "pred_x0", "pred_v"], help="training objective")
    parser.add_argument("--beta_schedule", type=str, default="cosine", choices=["linear", "cosine"], help="beta schedule")
    parser.add_argument("--finetune_last_n_layers", type=int, default=1, help="Number of DINOv2 layers to finetune (0 means freeze all)")
    
    # Add wandb arguments with defaults to True so no need to specify on command line
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="conditional-diffusion-film", help="W&B project name")  # 更新项目名
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    
    args = parser.parse_args()

    # 创建模型 - 无需修改，因为FiLM替换已在模型定义中完成
    model = ConditionalDiffusionModel(
        unet_dim=args.unet_dim,
        seq_length=args.max_tensor_len,
        finetune_last_n_layers=args.finetune_last_n_layers, 
        diffusion_kwargs={
            'timesteps': args.timesteps,
            'sampling_timesteps': args.sampling_timesteps,
            'objective': args.objective,
            'beta_schedule': args.beta_schedule,
            'channels': args.feature_dim,
            'dim_mults': tuple(map(int, args.unet_dim_mults.split(','))),
            'dropout': args.dropout,
            'self_condition': False,
            'learned_variance': False,
            'learned_sinusoidal_cond': False,
            'random_fourier_features': False,
            'attn_dim_head': 8,
            'attn_heads': 4,
        }
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"FiLM Model total size: {total_params/1e6:.2f}M, trainable: {trainable_params/1e6:.2f}M")
    print(f"DINOv2 trainable parameters: {encoder_trainable/1e6:.2f}M")

    print(f"Starting training with FiLM, data path: {args.data_path}")
    train_conditional_diffusion(
        model=model,
        data_path=args.data_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_tensor_len=args.max_tensor_len,
        feature_dim=args.feature_dim,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_ratio=args.eval_ratio,
        resume_path=args.resume,
        resume_optimizer=True,
        resume_scheduler=True,
        resume_epoch=True,
        seed=args.seed,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )

    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()

