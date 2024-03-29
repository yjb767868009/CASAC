import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging

from .AT import AT
from .BaseModel import BaseModel
from model.utils import model2gpu


class ATM(BaseModel):
    """
    Animation Transformer Model
    """

    def __init__(self, save_path, lr=0.0001):
        super().__init__()
        self.save_path = save_path
        self.lr = lr
        self.at = model2gpu(AT(save_path))
        self.at_optimizer = torch.optim.AdamW(self.at.parameters(), lr=self.lr)

        self.writer = SummaryWriter(os.path.join(self.save_path, 'log'))
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s  %(message)s',
                            filename=os.path.join(self.save_path, 'log.txt'))

    def train_init(self):
        self.at.train()

    def test_init(self):
        self.at.eval()

    def save(self):
        # Save Model
        torch.save(self.at.state_dict(), os.path.join(self.save_path, "at.pth"))
        # Save optimizer
        torch.save(self.at_optimizer.state_dict(), os.path.join(self.save_path, "at_optimizer.pth"))

    def load_param(self, load_path):
        # Load Model
        self.at.load_state_dict(torch.load(os.path.join(load_path, 'at.pth')))
        # Load optimizer
        self.at_optimizer.load_state_dict(torch.load(os.path.join(load_path, 'at_optimizer.pth')))

    def update_lr(self):
        self.lr /= 2
        for param_group in self.at_optimizer.param_groups:
            param_group['lr'] = self.lr

    def ep(self, data_iter, epoch, train):
        all_loss_list = []
        pose_loss_list = []
        inverse_pose_loss_list = []
        trajectory_pose_loss_list = []
        inverse_trajectory_pose_loss_list = []
        goal_loss_list = []
        contact_loss_list = []
        contact_hips_loss_list = []
        contact_right_wrist_loss_list = []
        contact_left_wrist_loss_list = []
        contact_right_ankle_loss_list = []
        contact_left_ankle_loss_list = []
        phase_loss_list = []
        loss_list = []

        for input, label in tqdm(data_iter, ncols=100):
            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()
            if train:
                self.at_optimizer.zero_grad()

            output = self.forward(input)

            all_loss = nn.MSELoss()(output.mean(dim=1), label[:, -1, :])
            pose_loss = nn.MSELoss()(output[:, :, 0:276].mean(dim=1), label[:, -1, 0:276])
            inverse_pose_loss = nn.MSELoss()(output[:, :, 276:345].mean(dim=1), label[:, -1, 276:345])
            trajectory_pose_loss = nn.MSELoss()(output[:, :, 345:422].mean(dim=1), label[:, -1, 345:422])
            inverse_trajectory_pose_loss = nn.MSELoss()(output[:, :, 422:450].mean(dim=1), label[:, -1, 422:450])
            goal_loss = nn.MSELoss()(output[:, :, 450:606].mean(dim=1), label[:, -1, 450:606])
            contact_loss = nn.MSELoss()(output[:, :, 606:611].mean(dim=1), label[:, -1, 606:611])
            contact_hips_loss = nn.MSELoss()(output[:, :, 606].mean(dim=1), label[:, -1, 606])
            contact_right_wrist_loss = nn.MSELoss()(output[:, :, 607].mean(dim=1), label[:, -1, 607])
            contact_left_wrist_loss = nn.MSELoss()(output[:, :, 608].mean(dim=1), label[:, -1, 608])
            contact_right_ankle_loss = nn.MSELoss()(output[:, :, 609].mean(dim=1), label[:, -1, 609])
            contact_left_ankle_loss = nn.MSELoss()(output[:, :, 610].mean(dim=1), label[:, -1, 610])
            phase_loss = nn.MSELoss()(output[:, :, 611:618].mean(dim=1), label[:, -1, 611:618])
            loss = pose_loss + inverse_pose_loss + trajectory_pose_loss + inverse_trajectory_pose_loss + goal_loss + contact_loss + phase_loss

            all_loss_list.append(all_loss.item())
            pose_loss_list.append(pose_loss.item())
            inverse_pose_loss_list.append(inverse_pose_loss.item())
            trajectory_pose_loss_list.append(trajectory_pose_loss.item())
            inverse_trajectory_pose_loss_list.append(inverse_trajectory_pose_loss.item())
            goal_loss_list.append(goal_loss.item())
            contact_loss_list.append(contact_loss.item())
            contact_hips_loss_list.append(contact_hips_loss.item())
            contact_right_wrist_loss_list.append(contact_right_wrist_loss.item())
            contact_left_wrist_loss_list.append(contact_left_wrist_loss.item())
            contact_right_ankle_loss_list.append(contact_right_ankle_loss.item())
            contact_left_ankle_loss_list.append(contact_left_ankle_loss.item())
            phase_loss_list.append(phase_loss.item())
            loss_list.append(loss.item())

            if train:
                loss.backward()
                self.at_optimizer.step()
        all_loss = np.asarray(all_loss_list).mean()
        pose_loss = np.asarray(pose_loss_list).mean()
        inverse_pose_loss = np.asarray(inverse_pose_loss_list).mean()
        trajectory_pose_loss = np.asarray(trajectory_pose_loss_list).mean()
        inverse_trajectory_pose_loss = np.asarray(inverse_trajectory_pose_loss_list).mean()
        goal_loss = np.asarray(goal_loss_list).mean()
        contact_loss = np.asarray(contact_loss_list).mean()
        contact_hips_loss = np.asarray(contact_hips_loss_list).mean()
        contact_right_wrist_loss = np.asarray(contact_right_wrist_loss_list).mean()
        contact_left_wrist_loss = np.asarray(contact_left_wrist_loss_list).mean()
        contact_right_ankle_loss = np.asarray(contact_right_ankle_loss_list).mean()
        contact_left_ankle_loss = np.asarray(contact_left_ankle_loss_list).mean()
        phase_loss = np.asarray(phase_loss_list).mean()
        loss = np.asarray(loss_list).mean()

        title = "Train" if train else "Test"
        self.writer.add_scalars('Loss/all_loss', {title: all_loss}, epoch)
        self.writer.add_scalars('Loss/pose_loss', {title: pose_loss}, epoch)
        self.writer.add_scalars('Loss/inverse_pose_loss', {title: inverse_pose_loss}, epoch)
        self.writer.add_scalars('Loss/trajectory_loss', {title: trajectory_pose_loss}, epoch)
        self.writer.add_scalars('Loss/inverse_trajectory_pose_loss', {title: inverse_trajectory_pose_loss}, epoch)
        self.writer.add_scalars('Loss/goal_loss', {title: goal_loss}, epoch)
        self.writer.add_scalars('Loss/contact_loss', {title: contact_loss}, epoch)
        self.writer.add_scalars('Loss/phase_loss', {title: phase_loss}, epoch)
        self.writer.add_scalars('Loss/loss', {title: loss}, epoch)

        self.writer.add_scalars('Contact/hips', {title: contact_hips_loss}, epoch)
        self.writer.add_scalars('Contact/right_wrist', {title: contact_right_wrist_loss}, epoch)
        self.writer.add_scalars('Contact/left_wrist', {title: contact_left_wrist_loss}, epoch)
        self.writer.add_scalars('Contact/right_ankle', {title: contact_right_ankle_loss}, epoch)
        self.writer.add_scalars('Contact/left_ankle', {title: contact_left_ankle_loss}, epoch)

        message = 'Epoch {} : '.format(epoch + 1) + title + \
                  'Loss = {:.5f} '.format(all_loss)
        logging.info(message)

    def view_attention(self, data_iter):
        index = 0
        for input, label in tqdm(data_iter, ncols=100):
            input = input[0].unsqueeze(0)
            if torch.cuda.is_available():
                input = input.cuda()
            self.at(input)
            index += 1
            if index > 20:
                break

    def forward(self, x):
        output = self.at(x)
        return output

    def __str__(self):
        return str(self.at)
