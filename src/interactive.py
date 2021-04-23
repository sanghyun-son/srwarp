import os
from os import path
import sys
import math
import argparse
import typing

import numpy as np
import imageio
from PIL import Image
import pyscreenshot
import cv2
import torch
from torch import hub
from torchvision import utils as vutils

from srwarp import grid
from srwarp import resize
from srwarp import warp
from srwarp import transform
from model.srwarp import baseline
from model.srwarp import lightwarp
from model.sr import edsr

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QPen
from PyQt5.QtGui import QBrush
from PyQt5.QtCore import Qt


def np2tensor(x: np.array) -> torch.Tensor:
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x)
    with torch.no_grad():
        while x.dim() < 4:
            x.unsqueeze_(0)

        x = x.float()
        x = x / 127.5 - 1

    return x

def tensor2np(x: torch.Tensor) -> np.array:
    with torch.no_grad():
        x = 127.5 * (x + 1)
        x.clamp_(min=0, max=255)
        x.round_()
        x = x.byte()
        x = x.squeeze(0)

    x = x.cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    x = np.ascontiguousarray(x)
    return x


class Interactive(QMainWindow):

    def __init__(self, app: QApplication) -> None:

        super().__init__()

        parser = argparse.ArgumentParser()
        parser.add_argument('--img', type=str, default='example/butterfly.png')
        parser.add_argument('--full', action='store_true')
        parser.add_argument('--load_m', action='store_true')
        parser.add_argument('--kernel_size', type=int, default=4)
        parser.add_argument('--pretrained', type=str)
        parser.add_argument('--no_residual', action='store_true')
        parser.add_argument('--no_depthwise', action='store_true')
        parser.add_argument('--backbone', type=str, default='mdsr')
        parser.add_argument('--record', action='store_true')
        cfg = parser.parse_args()

        self.setStyleSheet('background-color: white;')
        self.margin = 300
        img = Image.open(cfg.img)
        self.img = np.array(img)
        self.img_tensor = np2tensor(self.img).cuda()
        self.img_h = self.img.shape[0]
        self.img_w = self.img.shape[1]
        self.offset_h = self.margin
        self.offset_w = self.img_w + 2 * self.margin

        self.window_h = self.img_h + 2 * self.margin
        self.window_w = 2 * self.img_w + 3 * self.margin

        monitor_resolution = app.desktop().screenGeometry()
        self.screen_h = monitor_resolution.height()
        self.screen_w = monitor_resolution.width()

        self.screen_offset_h = (self.screen_h - self.window_h) // 2
        self.screen_offset_w = (self.screen_w - self.window_w) // 2

        self.setGeometry(
            self.screen_offset_w,
            self.screen_offset_h,
            self.window_w,
            self.window_h,
        )
        self.reset_cps()

        self.line_order = ('tl', 'tr', 'br', 'bl')
        self.grab = None
        self.shift = False

        self.inter = cv2.INTER_LINEAR
        self.backend = 'srwarp'

        model_class = baseline.SuperWarpF
        self.net = model_class(
            max_scale=4,
            backbone=cfg.backbone,
            residual=not cfg.no_residual,
            kernel_net=True,
            kernel_net_multi=True,
            kernel_size_up=3,
            kernel_depthwise=not cfg.no_depthwise,
            fill=255,
        )
        self.net.cuda()

        if cfg.pretrained is not None:
            print('Loading pre-trained model from {}...'.format(cfg.pretrained))
            state = torch.load(cfg.pretrained)
            state = state['model']
            self.net.load_state_dict(state, strict=False)
            self.net.build(self.img_tensor)

        self.backup = None
        self.backup_img = None
        self.record = cfg.record
        self.fidx = 0

        self.fix_transform = False

        # For debugging
        torch.set_printoptions(precision=3, linewidth=240, edgeitems=8, sci_mode=False)
        self.update()
        return

    def reset_cps(self) -> None:
        self.cps = {
            'tl': (0, 0),
            'tr': (0, self.img_w - 1),
            'bl': (self.img_h - 1, 0),
            'br': (self.img_h - 1, self.img_w - 1),
        }
        return

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key_Shift:
            self.shift = False

        return

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_1:
            self.backend = 'opencv'
            self.update()
        if e.key() == Qt.Key_2:
            self.backend = 'core'
            self.update()
        if e.key() == Qt.Key_3:
            self.backend = 'srwarp'
            self.update()

        if e.key() == Qt.Key_Escape:
            self.close()

        if e.key() == Qt.Key_Shift:
            self.shift = True

        if e.key() == Qt.Key_I:
            if self.inter == cv2.INTER_CUBIC:
                self.inter = cv2.INTER_NEAREST
            elif self.inter == cv2.INTER_NEAREST:
                self.inter = cv2.INTER_LINEAR
            else:
                self.inter = cv2.INTER_CUBIC

            self.update()
        elif e.key() == Qt.Key_F:
            self.fix_transform = not self.fix_transform
            self.update()
        elif e.key() == Qt.Key_R:
            self.reset_cps()
            self.update()

        return

    def mousePressEvent(self, e) -> None:
        is_left = e.buttons() & Qt.LeftButton
        if is_left:
            threshold = 20
            min_dist = 987654321
            for key, val in self.cps.items():
                y, x = val
                dy = e.y() - y - self.offset_h
                dx = e.x() - x - self.offset_w
                dist = dy ** 2 + dx ** 2
                if dist < min_dist:
                    min_dist = dist
                    self.grab = key

            if min_dist > threshold ** 2:
                self.grab = None

        return

    def get_matrix(self) -> torch.Tensor:
        points_from = np.array([
            [0, 0],
            [self.img_w - 1, 0],
            [0, self.img_h - 1],
            [self.img_w - 1, self.img_h - 1],
        ]).astype(np.float32)
        points_to = np.array([
            [self.cps['tl'][1], self.cps['tl'][0]],
            [self.cps['tr'][1], self.cps['tr'][0]],
            [self.cps['bl'][1], self.cps['bl'][0]],
            [self.cps['br'][1], self.cps['br'][0]],
        ]).astype(np.float32)
        m = cv2.getPerspectiveTransform(points_from, points_to)
        m = torch.Tensor(m)
        m = m.double()
        return m

    def mouseMoveEvent(self, e) -> None:
        if self.grab is None:
            return

        y_old, x_old = self.cps[self.grab]
        y_new = e.y() - self.offset_h
        x_new = e.x() - self.offset_w
        self.cps[self.grab] = (y_new, x_new)
        if self.shift:
            tb = self.grab[0]
            lr = self.grab[1]
            anchor = None
            for key, val in self.cps.items():
                if not (tb in key or lr in key):
                    anchor = val
                    break

            for key, val in self.cps.items():
                if key == self.grab:
                    continue

                if tb in key:
                    self.cps[key] = (y_new, anchor[1])

                if lr in key:
                    self.cps[key] = (anchor[0], x_new)

        is_convex = True
        #cross = None
        for i, pos in enumerate(self.line_order):
            y1, x1 = self.cps[pos]
            y2, x2 = self.cps[self.line_order[(i + 1) % 4]]
            y3, x3 = self.cps[self.line_order[(i + 2) % 4]]
            dx1 = x2 - x1
            dy1 = y2 - y1
            dx2 = x3 - x2
            dy2 = y3 - y2
            cross_new = dx1 * dy2 - dy1 * dx2
            if cross_new < 3000:
                is_convex = False
                break

        if not is_convex:
            self.cps[self.grab] = (y_old, x_old)

        self.update()

        if self.record:
            os.makedirs('record', exist_ok=True)
            screen = pyscreenshot.grab(
                bbox=(
                    self.screen_offset_w,
                    self.screen_offset_h,
                    self.screen_offset_w + self.window_w,
                    self.screen_offset_h + self.window_h,
                )
            )
            screen.save(path.join('record', f'{self.fidx:0>3}.bmp'))
            self.fidx += 1

        return

    def mouseReleaseEvent(self, e) -> None:
        if self.grab is not None:
            self.grab = None

        self.update()
        return

    @torch.no_grad()
    def paintEvent(self, e) -> None:
        qp = QPainter()
        qp.begin(self)

        #if self.grab is None:
        if True:
            if self.inter == cv2.INTER_NEAREST:
                inter_method = 'Nearest'
            elif self.inter == cv2.INTER_LINEAR:
                inter_method = 'Bilinear'
            elif self.inter == cv2.INTER_CUBIC:
                inter_method = 'Bicubic'

            self.setWindowTitle(
                f'Interpolation: {inter_method} / backend: {self.backend}',
            )

            qimg = QImage(
                self.img,
                self.img_w,
                self.img_h,
                3 * self.img_w,
                QImage.Format_RGB888,
            )
            qpix = QPixmap(qimg)
            qp.drawPixmap(self.margin, self.margin, self.img_w, self.img_h, qpix)

            m = self.get_matrix()
            m, sizes, offsets = transform.compensate_matrix(self.img_tensor, m)
            h_new, w_new = sizes
            y_min, x_min = offsets
            if self.backend == 'opencv':
                y = cv2.warpPerspective(
                    self.img, m.numpy(), (w_new, h_new), flags=self.inter,
                )
            elif self.backend == 'core':
                if self.fix_transform:
                    y = resize.imresize(self.img_tensor, scale=2)
                else:
                    y = warp.warp_by_function(
                        self.img_tensor,
                        m,
                        f_inverse=False,
                        sizes=(h_new, w_new),
                        adaptive_grid=(inter_method.lower() != 'bicubic'),
                        fill=255,
                    )
                y = tensor2np(y)
                self.backup_img = y
            elif self.backend == 'srwarp':
                y, mask = self.net(
                    self.img_tensor,
                    m,
                    sizes=(h_new, w_new),
                )
                y = tensor2np(y)
                self.backup_img = y

            qimg_warp = QImage(y, w_new, h_new, 3 * w_new, QImage.Format_RGB888)
            qpix_warp = QPixmap(qimg_warp)
            qp.drawPixmap(
                self.offset_w - x_min,
                self.offset_h - y_min,
                w_new,
                h_new,
                qpix_warp,
            )

        center_y = self.offset_h + self.img_h // 2
        center_x = self.offset_w + self.img_w // 2

        pen_blue = QPen(Qt.blue, 5)
        pen_white = QPen(Qt.white, 10)
        text_size = 20
        for key, val in self.cps.items():
            y, x = val
            y = y + self.offset_h
            x = x + self.offset_w
            qp.setPen(pen_blue)
            qp.drawPoint(x, y)
            qp.setPen(pen_white)
            dy = y - center_y
            dx = x - center_x
            dl = math.sqrt(dy ** 2 + dx ** 2) / 10
            qp.drawText(
                int(x + (dx / dl) - text_size / 2),
                int(y + (dy / dl) - text_size / 2),
                text_size,
                text_size,
                int(Qt.AlignCenter),
                key,
            )

        qp.end()
        return


def main() -> None:
    app = QApplication(sys.argv)
    sess = Interactive(app)
    sess.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()