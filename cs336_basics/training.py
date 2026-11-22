import cs336_basics.modules as m
import torch
import numpy as np
import wandb
from pathlib import Path
from cs336_basics.config import *

lm = m.TransformerLM(d_model, num_heads, d_ff, vocab_size, context_len, num_layers, theta, device=device)
optimizer = m.AdamW(lm.parameters(), max_lr, weight_decay, eps, betas, device=device)

# print(lm.named_parameters)
# for name, p in lm.named_parameters():
#     print(p.device)
# exit()

if load_from:
    steps = m.load_checkpoint(load_from, lm, optimizer)
else:
    steps = 0

Path(ckpt_folder).mkdir(parents=True, exist_ok=True)

training_data = np.memmap(train_token_file, dtype='<u2', mode='r')
validation_data = np.memmap(validation_token_file, dtype='<u2', mode='r')

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="jjjjjtyu",
    # Set the wandb project where this run will be logged.
    project="tmp1",
    # Track hyperparameters and run metadata.
    config=config,
)

for step in range(steps, total_steps):
    x, truth = m.load_data(training_data, batch_size, context_len, device)
    lm.zero_grad()
    logits = lm(x)
    loss = m.cross_entropy(logits, truth)
    print(f"step: {step}, loss: {loss}")
   
    loss.backward()
    lr = m.get_schedule(step, max_lr, min_lr, warmup_itrs, cosine_itrs)
    for g in optimizer.param_groups:
        g['lr'] = lr
    m.gradient_clipping(lm.parameters(), max_l2_norm)
    run.log({"loss": loss, "lr": lr})
    optimizer.step()
    if step % ckpt_intervals == 0 and step != 0:
        m.save_checkpoint(lm, optimizer, step, ckpt_folder + f"/{step}.ckpt")
    if step % ckpt_intervals == 0 and step != 0:
        with torch.no_grad():
            x, truth = m.load_data(validation_data, 100, context_len, device)
            pred = lm(x)
            validation_loss = m.cross_entropy(pred, truth)
            print(f"validation loss: {validation_loss}")

