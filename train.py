from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm
from model import GPT
from config import GPTConfig, TrainingConfig
from functools import partial
import time
import math
import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from dataset import Task

def save_checkpoint(model, model_args, iter_num, best_val_loss):
    checkpoint = {
        "model": model.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

def forward_hook_for_causal_attention_block(writer: SummaryWriter, block_idx: int, module: "nn.Module", x):
    writer.add_histogram(f'attention_block_input_length_block_{block_idx}', x[0].shape[1])
    return x

train_config = TrainingConfig(
    compile=False, 
    device="mps",
    train_iters=1000,
    eval_interval=50,
    batch_size=32)
out_dir = Path("out/")
writer = SummaryWriter(log_dir=out_dir / "logs" / datetime.now().strftime('%Y-%m-%d/%H:%M:%s'), flush_secs=2)
resume = False

tokens_per_iter = (
    train_config.gradient_accumulation_steps
    * train_config.batch_size
    * GPTConfig.block_size
)
print("Tokens per iteration: ", tokens_per_iter)

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

ctx = torch.autocast(train_config.device, dtype=torch.bfloat16)

model_args = dict(
    n_layer=GPTConfig.n_layer,
    n_head=GPTConfig.n_head,
    n_embed=GPTConfig.n_embed,
    block_size=GPTConfig.block_size,
    bias=GPTConfig.bias,
    vocab_size=GPTConfig.vocab_size,
    dropout=GPTConfig.dropout,
)

iter_batches = partial(
    Task.iter_batches,
    batch_size=train_config.batch_size,
    max_seq_len=GPTConfig.block_size,
    device=train_config.device,
    num_workers=0,
    chunk_ratios=np.arange(0.1, 1.1, 0.1),
    increase_chunk_ratio_every_n_shards=49
)

best_val_loss = 1e9
writer_hook = partial(forward_hook_for_causal_attention_block, writer)
if resume:
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=train_config.device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf, writer_hook)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    best_val_loss = checkpoint["best_val_loss"]

    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    print("Loaded checkpoint")
else:
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, writer_hook).to(train_config.device)

scaler = torch.GradScaler(enabled=(train_config.dtype == "float16"))
optimizer = model.configure_optimizers(
    train_config.weight_decay,
    train_config.learning_rate,
    (train_config.beta1, train_config.beta2),
    train_config.device,
)

if train_config.compile:
    model = torch.compile(model)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(train_config.eval_iters)
        batch_iter = iter_batches(split=split)
        for k in range(train_config.eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < train_config.warmup_iters:
        return train_config.learning_rate * (it + 1) / (train_config.warmup_iters + 1)
    if it > train_config.lr_decay_iters:
        return train_config.min_lr
    decay_ratio = (it - train_config.warmup_iters) / (
        train_config.lr_decay_iters - train_config.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (
        train_config.learning_rate - train_config.min_lr
    )


params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print("Number of params:", params)

train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)

iter_num = 0
t0 = time.time()
for iter_num in tqdm(range(train_config.train_iters)):
    lr = get_lr(iter_num) if train_config.decay_lr else train_config.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % train_config.eval_interval == 0:
        losses = estimate_loss()
        writer.add_scalar("train_loss", losses["train"], iter_num)
        writer.add_scalar("val_loss", losses["val"], iter_num)
        writer.add_scalar("lr", lr, iter_num)

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            save_checkpoint(model, model_args, iter_num, best_val_loss)

    for micro_step in range(train_config.gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / train_config.gradient_accumulation_steps
            writer.add_scalar("step_train_loss", loss, iter_num * train_config.gradient_accumulation_steps + micro_step)
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if train_config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

writer.close()
