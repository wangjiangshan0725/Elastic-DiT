import torch

from .base_pipeline import BasePipeline


def global_width_budget_loss(prob_width_list, probs_list, target=0.5, eps=1e-6):
    """
    prob_width_list: list length = num_blocks
        each element: [B, 4] or [1, 4], probabilities that sum to 1
    probs_list: list length = num_blocks
        each element: [B] or [B, 1], gate probability per block
    target: desired global mean ratio (e.g., 0.5)
    """

    # [num_blocks, B, 4]
    probs_width = torch.stack(prob_width_list, dim=0)

    # [num_blocks, B]
    probs_gate = torch.stack(
        [p.squeeze(-1) for p in probs_list], dim=0
    )

    # mask: 1 if prob > 0.5 else 0
    mask = (probs_gate > 0.5).float()

    ratios = probs_width.new_tensor([0.25, 0.5, 0.75, 1.0])

    # expected width ratio per block per sample: [num_blocks, B]
    exp_ratio = (probs_width * ratios).sum(dim=-1)

    # masked mean (avoid division by zero)
    masked_sum = (exp_ratio * mask).sum()
    mask_count = mask.sum().clamp_min(eps)

    mean_ratio = masked_sum / mask_count

    return (mean_ratio - target) ** 2

def map_0_1_to_range(x: float, low: float, high: float) -> float:
    """
    将 [0, 1] 区间的 x 映射到 [low, high]，
    且 0 -> high, 1 -> low（随 x 增大线性减小）
    """
    if not (0.0 <= x <= 1.0):
        raise ValueError("x must be in [0, 1]")
    return high + (low - high) * x

def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, info_list = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    print("!!pipe.loss_type!!!!",pipe.loss_type)
    if pipe.loss_type == 'skip_block+adaptive_linear':
        probs_list, prob_width_list = info_list
        probs = torch.cat(probs_list, dim=0).squeeze()  
        loss_probs = (probs.mean() - map_0_1_to_range(timestep / len(pipe.scheduler.timesteps), pipe.thre_low, pipe.thre_high)) ** 2
        
        count_gt_05 = sum((p > 0.5).item() for p in probs_list)
        print("num probs > 0.5:", count_gt_05)
        print(probs)
        print("target_width:", pipe.target_width)
        loss_probs_width = global_width_budget_loss(
            prob_width_list,
            probs_list,
            target=pipe.target_width,
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        # loss = loss * pipe.scheduler.training_weight(timestep)
        return loss + loss_probs + loss_probs_width
    else:
        probs = torch.cat(info_list[0], dim=0).squeeze()  
        loss_probs = (probs.mean() - map_0_1_to_range(timestep / len(pipe.scheduler.timesteps), pipe.thre_low, pipe.thre_high)) ** 2
        # print("timestep_id[0] / len(pipe.scheduler.timesteps)",timestep_id[0] / len(pipe.scheduler.timesteps))
        count_gt_05 = sum((p > 0.5).item() for p in info_list[0])
        print("num probs > 0.5:", count_gt_05)
        print(probs)
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        # loss = loss * pipe.scheduler.training_weight(timestep)
        return loss + loss_probs


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips  # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
