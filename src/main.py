import sys
import importlib

from trainer import base_trainer
from config import get_config
from misc import logger

import torch
import tqdm

def main() -> None:
    cfg = get_config.parse()
    try:
        with logger.Logger(cfg) as l:
            t = base_trainer.get_trainer(cfg, l)
            if cfg.test_only:
                t.evaluation()
            else:
                tq = tqdm.trange(t.begin_epoch, cfg.epochs, ncols=80)
                for epoch in tq:
                    tq.set_description(
                        'Epoch {}/{}'.format(epoch + 1, cfg.epochs),
                    )
                    t.fit()
                    if (epoch + 1) % cfg.test_period == 0:
                        t.evaluation()

                    t.at_epoch_end()

            t.misc.join_background()
    except KeyboardInterrupt:
        print('!!!Terminate the program!!!')
        exit()

    return

if __name__ == '__main__':
    main()
