import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from torch import nn
from torch.nn import functional as F

from .bfp.projector_manager import ProjectorManager, add_parser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # Consistency Regularization Weight
    parser.add_argument('--reg_weight', type=float, default=0.1)

    # Stable Model parameters
    parser.add_argument('--stable_model_update_freq', type=float, default=0.70)
    parser.add_argument('--stable_model_alpha', type=float, default=0.999)

    # Plastic Model Parameters
    parser.add_argument('--plastic_model_update_freq', type=float, default=0.90)
    parser.add_argument('--plastic_model_alpha', type=float, default=0.999)

    # BFP
    parser.add_argument("--skip_first", type=str2bool, default=False,
                        help="If true, BFP loss is not used during the first task")
    parser.add_argument("--proj_to", type=str, default="clser", 
                        choices=['stable', 'plastic', 'clser', 'old'],
                        help="To the output of which model the BFP is projected. ")
    parser.add_argument("--eval_mode", action="store_true", 
                        help="If set, plastic model and stable model will be used in eval mode")

    parser = add_parser(parser)

    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class CLSERBFP(ContinualModel):
    NAME = 'clserbfp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(CLSERBFP, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        # Initialize plastic and stable model
        self.plastic_model = deepcopy(self.net).to(self.device)
        self.stable_model = deepcopy(self.net).to(self.device)
        if self.args.eval_mode:
            self.plastic_model.eval()
            self.stable_model.eval()

        # set regularization weight
        self.reg_weight = args.reg_weight
        # set parameters for plastic model
        self.plastic_model_update_freq = args.plastic_model_update_freq
        self.plastic_model_alpha = args.plastic_model_alpha
        # set parameters for stable model
        self.stable_model_update_freq = args.stable_model_update_freq
        self.stable_model_alpha = args.stable_model_alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = -1
        self.global_step = 0

        # Backward feature projection
        self.projector_manager = ProjectorManager(self.args, self.net.net_channels, self.device)

        # Old network checkpoint used for proj2old
        if self.args.proj_to == "old":
            self.args.skip_first = True
            self.old_net = None

    def begin_task(self, dataset):
        self.current_task += 1

        # Not sure whether we should reset the projector though
        self.projector_manager.begin_task(dataset, self.current_task, 0) 

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        loss = 0

        if not self.buffer.is_empty():

            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            stable_feats, stable_model_logits = self.stable_model.extract_features(buf_inputs)
            plastic_feats, plastic_model_logits = self.plastic_model.extract_features(buf_inputs)

            stable_model_prob = F.softmax(stable_model_logits, 1)
            plastic_model_prob = F.softmax(plastic_model_logits, 1)

            label_mask = F.one_hot(buf_labels, num_classes=stable_model_logits.shape[-1]) > 0
            sel_idx = stable_model_prob[label_mask] > plastic_model_prob[label_mask]
            sel_idx = sel_idx.unsqueeze(1)

            ema_logits = torch.where(
                sel_idx,
                stable_model_logits,
                plastic_model_logits,
            )

            working_feats, working_logits = self.plastic_model.extract_features(buf_inputs)

            l_cons = torch.mean(self.consistency_loss(working_logits, ema_logits.detach()))
            l_reg = self.args.reg_weight * l_cons
            loss += l_reg

            # Log values
            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/l_cons', l_cons.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/l_reg', l_reg.item(), self.iteration)

            # Backward feature projection
            if self.projector_manager.bfp_flag:
                if self.args.skip_first and self.current_task == 0:
                    bfp_loss_all = 0.0
                else:
                    if self.args.proj_to == "clser":
                        ema_feats = [torch.where(
                            sel_idx.unsqueeze(1).unsqueeze(1), sf, pf
                        ).detach() for sf, pf in zip(stable_feats, plastic_feats)]
                    elif self.args.proj_to == "stable":
                        ema_feats = [f.detach() for f in stable_feats]
                    elif self.args.proj_to == "plastic":
                        ema_feats = [f.detach() for f in plastic_feats]
                    elif self.args.proj_to == "old":
                        with torch.no_grad():
                            self.old_net.eval()
                            _, ema_feats = self.old_net.forward_all_layers(buf_inputs)
                    
                    mask_old = torch.ones_like(buf_labels)
                    mask_new = torch.zeros_like(buf_labels)
                    bfp_loss_all, bfp_loss_dict = self.projector_manager.compute_loss(
                        working_feats, ema_feats, mask_new, mask_old)
                loss += bfp_loss_all

                if hasattr(self, 'writer'):
                    self.writer.add_scalar(f'Task {self.current_task}/l_bfp', l_cons.item(), self.iteration)

            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))


        outputs = self.net(inputs)
        ce_loss = self.loss(outputs, labels)
        loss += ce_loss

        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)

        self.opt.zero_grad()
        self.projector_manager.before_backward()

        loss.backward()
        self.opt.step()
        self.projector_manager.step()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size],
        )

        # Update the ema model
        self.global_step += 1
        if torch.rand(1) < self.plastic_model_update_freq:
            self.update_plastic_model_variables()

        if torch.rand(1) < self.stable_model_update_freq:
            self.update_stable_model_variables()

        return loss.item()

    def update_plastic_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.plastic_model_alpha)
        for ema_param, param in zip(self.plastic_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_stable_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1),  self.stable_model_alpha)
        for ema_param, param in zip(self.stable_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def end_task(self, dataset):
        self.old_net = deepcopy(self.net)
        self.old_net.eval()

        self.projector_manager.end_task(dataset, self.net)
