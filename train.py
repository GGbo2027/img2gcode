import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from tqdm import tqdm
import glob
from contextlib import nullcontext


class TensorImageDataset(Dataset):
    def __init__(
        self,
        data_path,
        image_size=64,
        max_tensor_len=165,
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
        
        
        image = np.load(img_path)


        if image.ndim == 3 and image.shape[0] not in [1, 3]:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 2:  
            image = np.expand_dims(image, 0)

        image = image.astype(np.float32)

        image = image / 255.0

        image = (image - 0.5) / 0.5

        image = torch.from_numpy(image).float()

        
        if image.dim() == 3:
            _, h, w = image.shape
            if h != self.image_size or w != self.image_size:
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), 
                    size=(self.image_size, self.image_size), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
   
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
                print(f"文本文件读取也失败: {str(e2)}")
            
                tensor_data = np.zeros((0, self.feature_dim))

 
        if tensor_data.shape[0] > self.max_tensor_len:
      
            tensor_data = tensor_data[:self.max_tensor_len]
        
     
        original_length = tensor_data.shape[0]
        

        padded_tensor = np.zeros((self.max_tensor_len, self.feature_dim))
        padded_tensor[:original_length] = tensor_data
        
  
        mask = np.zeros(self.max_tensor_len)
        mask[:original_length] = 1.0
        
        return {
            'image': image, 
            'tensor': torch.from_numpy(padded_tensor).float(),
            'mask': torch.from_numpy(mask).float(),
            'length': original_length,
            'path': sample['folder']  
        }


import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm   
from .diffusion import UViT2DTensor, TensorDiffusion  



def visualize_predictions(model, data_loader, device, epoch, save_dir, debug: bool = False):
    viz_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    batch = next(iter(data_loader))
    images       = batch["image"].to(device)            # [B,C,H,W]
    true_tensors = batch["tensor"].cpu().numpy()        # [B,T_max,3]
    true_lengths = batch["length"].cpu().numpy()        # [B]
    num_samples  = min(4, len(images))

    with torch.no_grad():

        pred_out = model.p_sample_loop(images[:num_samples])

    # 拆包
    if isinstance(pred_out, tuple) and len(pred_out) == 2:
        pred_tensors, pred_lengths = pred_out
    else:                
        pred_tensors, pred_lengths = pred_out, None


    pred_list, len_list = [], []

    if isinstance(pred_tensors, (list, tuple)):
        pred_list = [p.cpu().numpy() for p in pred_tensors]
    else:
        # Tensor → [N, L, 3] 或 [L,3]
        if pred_tensors.dim() == 2:        # [L,3]，
            pred_list = [pred_tensors.cpu().numpy()]
        else:                              # [N, L_max,3]
            pred_list = [p.cpu().numpy() for p in pred_tensors]

    if pred_lengths is None:
        len_list = [p.shape[0] for p in pred_list]
    else:
        if isinstance(pred_lengths, torch.Tensor):
            pred_lengths = pred_lengths.cpu().tolist()
        elif isinstance(pred_lengths, np.ndarray):
            pred_lengths = pred_lengths.tolist()
        len_list = [int(l) for l in pred_lengths]


    if debug:
        print(f"[DEBUG] true_tensors : {true_tensors.shape}")
        print(f"[DEBUG] #pred samples : {len(pred_list)}")
        print(f"[DEBUG] true_lengths  : {true_lengths[:num_samples]}")
        print(f"[DEBUG] pred_lengths  : {len_list}")


    def save_and_close(fig, path):
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if debug:
            print(f"[DEBUG] saved → {path}")
        plt.close(fig)

    for i in range(num_samples):
        p_len = len_list[i]
        if p_len == 0:
            if debug:
                print(f"[DEBUG] skip sample {i} (pred_len==0)")
            continue

        fig = plt.figure(figsize=(12, 10))

        # 1️⃣ input_img
        ax = fig.add_subplot(2, 2, 1)
        img = (images[i].cpu().numpy().transpose(1, 2, 0) + 1) / 2   # [-1,1]→[0,1]
        ax.imshow(img); ax.set_title("Input Image"); ax.axis("off")

        # 2️⃣ true_tensor
        t_len  = int(true_lengths[i])
        true   = true_tensors[i, :t_len]
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(true[:, 0], true[:, 1], c=true[:, 2],
                cmap="viridis", s=20)
        ax2.set_title(f"True Tensor (Len={t_len})")
        #ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

        # 3️⃣ pre_tensor
        pred = pred_list[i][:p_len]
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(pred[:, 0], pred[:, 1], c=pred[:, 2],
                cmap="viridis", s=20)
        ax3.set_title(f"Predicted Tensor (Len={p_len})")
        #ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)

        # 4️⃣ comparison
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.scatter(true[:, 0], true[:, 1], c="blue",  s=20, alpha=0.5, label="True")
        ax4.scatter(pred[:, 0], pred[:, 1], c="red",   s=20, alpha=0.5, label="Pred")
        ax4.set_title("Comparison")
        #ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
        ax4.legend()

        save_and_close(fig, os.path.join(viz_dir, f"epoch_{epoch}_sample_{i}.png"))
    
    print(f"Visualizations saved to {viz_dir}")


   

def plot_training_history(history, save_dir):

    train_epochs = [x['epoch'] + x['step']/1000. for x in history['train_loss']]
    train_losses = [x['loss'] for x in history['train_loss']]


    val_epochs   = [x['epoch'] for x in history['val_loss']]
    val_losses   = [x['loss']  for x in history['val_loss']]


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




def train_tensor_diffusion(
    model: TensorDiffusion,
    data_path: str,
    save_dir: str = "./checkpoints",
    batch_size: int = 64,
    image_size: int = 64,
    max_tensor_len: int = 140,
    feature_dim: int = 3,
    num_workers: int = 12,
    epochs: int = 200,
    lr: float = 1e-5,
    device: str = "cuda",
    log_every: int = 10,
    save_every: int = 10,
    eval_ratio: float = 0.1,
    resume_path: str | None = None,
    resume_optimizer: bool = True,
    resume_scheduler: bool = True,
    resume_epoch: bool = True,
    mixed_precision: bool = True,
    seed: int = 42,
):
    """训练 Tensor-Diffusion 模型"""

    torch.manual_seed(seed)
    np.random.seed(seed)

  
    os.makedirs(save_dir, exist_ok=True)

   
    model = model.to(device)

 
    if mixed_precision and (not torch.cuda.is_available() or device.startswith("cpu")):
        print("CUDA is not usable, turn off AMP")
        mixed_precision = False


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

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr/10
    )
    scaler = GradScaler() if mixed_precision else None

    history = {"train_loss": [], "train_mse": [], "val_loss": [], "val_mse": []}

    # 8) Resume
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if resume_optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if resume_scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if resume_epoch and "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if mixed_precision and scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "history" in ckpt:
            history = ckpt["history"]
        print(f" From {resume_path} resume training, start epoch: {start_epoch}")
 

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        pbar = tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}", leave=False
        )

        for step, batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            images = batch["image"].to(device).contiguous()
            targets = batch["tensor"].to(device).contiguous()
            amp_ctx = autocast() if mixed_precision else nullcontext()
            with amp_ctx:
                out = model(img_tensor=images, target_tensor=targets)
            loss = out["loss"]
            mse = out["tensor_mse"].item()

            # backward
            if mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_mse += mse
            pbar.set_postfix({
                "Loss": f"{running_loss/(step+1):.4f}",
                "MSE": f"{running_mse/(step+1):.4f}",
            })
            if step % log_every == 0:
                history["train_loss"].append({
                    "epoch": epoch,
                    "step": step,
                    "loss": loss.item()
                })
                history["train_mse"].append({
                    "epoch": epoch,
                    "step": step,
                    "mse": mse
                })

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        avg_train_mse = running_mse / len(train_loader)
        print(f"Epoch {epoch+1} ▶ train_loss={avg_train_loss:.4f} | train_mse={avg_train_mse:.4f}")

        # 验证
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_mse = 0.0
            amp_ctx = autocast() if mixed_precision else nullcontext()
            with torch.no_grad(), amp_ctx:
                for batch in val_loader:
                    imgs = batch["image"].to(device).contiguous()
                    tens = batch["tensor"].to(device).contiguous()
                    out = model(img_tensor=imgs, target_tensor=tens)
                    loss = out["loss"]
                    mse = out["tensor_mse"].item()
                    val_loss += loss.item()
                    val_mse += mse
            val_loss /= len(val_loader)
            val_mse /= len(val_loader)
            history["val_loss"].append({
                "epoch": epoch,
                "loss": val_loss
            })
            history["val_mse"].append({
                "epoch": epoch,
                "mse": val_mse
            })
            print(f"Validation ▶ val_loss={val_loss:.4f} | val_mse={val_mse:.4f}")
            visualize_predictions(
                model=model,
                data_loader=val_loader,
                device=device,
                epoch=epoch+1,
                save_dir=save_dir,
                debug=False
            )

        # Checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
            }
            if scaler:
                ckpt["scaler_state_dict"] = scaler.state_dict()
            torch.save(ckpt, os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))
            torch.save(ckpt, os.path.join(save_dir, "model_latest.pt"))
            print(" Checkpoint saved")

    # 训练结束
    plot_training_history(history, save_dir)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tensor-Diffusion with two-phase strategy")
    parser.add_argument("--data_path", type=str, default="./filtered_dataset")
    parser.add_argument("--save_dir",  type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--lr",         type=float, default=5e-6)
    parser.add_argument("--resume",     type=str,  default=None)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--no_amp",     action="store_true", help="Disable AMP")
    args = parser.parse_args()

 
    image_size     = 64
    max_tensor_len = 145
    feature_dim    = 3

    

    uvit = UViT2DTensor(
        dim=128,
        max_tensor_len=max_tensor_len,
        feature_dim=feature_dim,
        vit_depth=8
    )

    model = TensorDiffusion(
        model=uvit,
        image_size=image_size,
        max_tensor_len=max_tensor_len,
        feature_dim=feature_dim,
        channels=3, 
        pred_objective="eps",
    )


    trained_model, history = train_tensor_diffusion(
        model=model,
        data_path=args.data_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        max_tensor_len=max_tensor_len,
        feature_dim=feature_dim,
        epochs=args.epochs,
        lr=args.lr,
        device="cuda" if torch.cuda.is_available() else "cpu",
        resume_path=args.resume,
        mixed_precision=not args.no_amp,
        seed=args.seed,

    )

    if history.get("train"):
        print(f"Final train ▶ {history['train'][-1]:.4f}")
    if history.get("val"):
        print(f"Final val ▶ {history['val'][-1]:.4f}")









