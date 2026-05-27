# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
from collections.abc import Iterable

import torch

from .util import lr_sched, misc


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}')
    )
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    accum_iter = args.accum_iter

    # optimizer.zero_grad()
    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (samples, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=True):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(
                f'[BAD LOSS] epoch={epoch} step={data_iter_step} loss={loss_value}'
            )
            print('shape:', tuple(samples.shape))
            finite_mask = torch.isfinite(samples)
            if finite_mask.any():
                print('input min:', samples[finite_mask].min().item())
                print('input max:', samples[finite_mask].max().item())
                print('input mean:', samples[finite_mask].mean().item())
                print(
                    'input std:',
                    samples[finite_mask].std(unbiased=False).item(),
                )

            torch.save(
                {
                    'epoch': epoch,
                    'step': data_iter_step,
                    'samples': samples.detach().cpu(),
                },
                f'debug_bad_loss_e{epoch}_s{data_iter_step}.pt',
            )

            optimizer.zero_grad(set_to_none=True)
            continue

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            # optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000
            )
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# import math
# import sys
# from typing import Iterable
# import os
# import torch

# from .util import misc
# from .util import lr_sched


# def train_one_epoch(model: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler,
#                     log_writer=None,
#                     args=None):
#     model.train(True)
#     metric_logger = misc.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 20

#     accum_iter = args.accum_iter
#     optimizer.zero_grad()

#     if log_writer is not None:
#         print('log_dir: {}'.format(log_writer.log_dir))

#     # 先强行关 AMP 做排查；如果排查完没问题再开回去
#     use_amp = False

#     for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         # 兼容不同 dataset 返回格式
#         if isinstance(batch, (tuple, list)):
#             samples = batch[0]
#             meta = batch[1:]
#         else:
#             samples = batch
#             meta = None

#         if data_iter_step % accum_iter == 0:
#             lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

#         samples = samples.to(device, non_blocking=True).float()

#         # ---------- 输入检查 ----------
#         finite_mask = torch.isfinite(samples)
#         if not finite_mask.all():
#             print(f"[BAD INPUT] epoch={epoch} step={data_iter_step}")
#             print("shape:", tuple(samples.shape))
#             print("nan_count:", torch.isnan(samples).sum().item())
#             print("inf_count:", torch.isinf(samples).sum().item())
#             if finite_mask.any():
#                 print("min:", samples[finite_mask].min().item())
#                 print("max:", samples[finite_mask].max().item())
#                 print("mean:", samples[finite_mask].mean().item())
#                 print("std:", samples[finite_mask].std(unbiased=False).item())
#             torch.save(
#                 {
#                     "epoch": epoch,
#                     "step": data_iter_step,
#                     "samples": samples.detach().cpu(),
#                     "meta": meta,
#                 },
#                 f"debug_bad_input_e{epoch}_s{data_iter_step}.pt"
#             )
#             raise ValueError("Input contains NaN/Inf before model forward")

#         # 看看是不是极端值 / 常数块
#         smin = samples.min().item()
#         smax = samples.max().item()
#         smean = samples.mean().item()
#         sstd = samples.std(unbiased=False).item()

#         if data_iter_step % 20 == 0:
#             print(f"[DEBUG INPUT] epoch={epoch} step={data_iter_step} "
#                   f"min={smin:.6f} max={smax:.6f} mean={smean:.6f} std={sstd:.6f}")

#         # ---------- forward ----------
#         try:
#             if use_amp:
#                 with torch.amp.autocast("cuda", enabled=True):
#                     loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)
#             else:
#                 loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)
#         except Exception as e:
#             print(f"[FORWARD EXCEPTION] epoch={epoch} step={data_iter_step}: {repr(e)}")
#             torch.save(
#                 {
#                     "epoch": epoch,
#                     "step": data_iter_step,
#                     "samples": samples.detach().cpu(),
#                     "meta": meta,
#                 },
#                 f"debug_forward_exception_e{epoch}_s{data_iter_step}.pt"
#             )
#             raise

#         # ---------- loss 检查 ----------
#         loss_value = loss.item()
#         if not math.isfinite(loss_value):
#             print(f"[BAD LOSS] epoch={epoch} step={data_iter_step}")
#             print("loss =", loss_value)
#             print(f"input min={smin:.6f} max={smax:.6f} mean={smean:.6f} std={sstd:.6f}")

#             # 进一步检查 pred / mask
#             if torch.is_tensor(pred):
#                 pred_finite = torch.isfinite(pred)
#                 print("[pred] shape:", tuple(pred.shape))
#                 print("[pred] nan_count:", torch.isnan(pred).sum().item())
#                 print("[pred] inf_count:", torch.isinf(pred).sum().item())
#                 if pred_finite.any():
#                     print("[pred] min:", pred[pred_finite].min().item())
#                     print("[pred] max:", pred[pred_finite].max().item())

#             if torch.is_tensor(mask):
#                 mask_finite = torch.isfinite(mask)
#                 print("[mask] shape:", tuple(mask.shape))
#                 print("[mask] nan_count:", torch.isnan(mask).sum().item())
#                 print("[mask] inf_count:", torch.isinf(mask).sum().item())
#                 if mask_finite.any():
#                     print("[mask] min:", mask[mask_finite].min().item())
#                     print("[mask] max:", mask[mask_finite].max().item())

#             torch.save(
#                 {
#                     "epoch": epoch,
#                     "step": data_iter_step,
#                     "samples": samples.detach().cpu(),
#                     "pred": pred.detach().cpu() if torch.is_tensor(pred) else pred,
#                     "mask": mask.detach().cpu() if torch.is_tensor(mask) else mask,
#                     "meta": meta,
#                 },
#                 f"debug_bad_loss_e{epoch}_s{data_iter_step}.pt"
#             )
#             raise ValueError("Loss became NaN/Inf")

#         # ---------- backward ----------
#         loss = loss / accum_iter
#         loss_scaler(loss, optimizer, parameters=model.parameters(),
#                     update_grad=(data_iter_step + 1) % accum_iter == 0)

#         if (data_iter_step + 1) % accum_iter == 0:
#             optimizer.zero_grad()

#         torch.cuda.synchronize()

#         metric_logger.update(loss=loss_value)
#         lr = optimizer.param_groups[0]["lr"]
#         metric_logger.update(lr=lr)

#         loss_value_reduce = misc.all_reduce_mean(loss_value)
#         if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
#             epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
#             log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
#             log_writer.add_scalar('lr', lr, epoch_1000x)

#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
