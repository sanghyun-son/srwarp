def set_template(template, cfg):
    if 'srwarp-all' in template:
        cfg.model = 'srwarp.baseline'
        cfg.residual = True
        cfg.kernel_size_up = 3
        cfg.kernel_net = True
        cfg.kernel_net_multi = True
        cfg.kernel_depthwise = True

    if 'down' in template:
        cfg.scale = 2
        cfg.patch = 64
        cfg.dtrain = ['downsampler.unpaired']
        cfg.augmentation = ''
        cfg.unpaired_hr = '../dataset/DIV2K/patch_gradient/crop_filtered_odd'
        if 'video' in template:
            cfg.unpaired_lr = '../experiment/jpeg/q20_test/patch_gradient/crop_seoul_moon_filtered'
        else:
            cfg.unpaired_lr = '../dataset/DIV2K/patch_gradient_gaussian/crop_lr_{}_filtered_even'.format(cfg.degradation_test)
            cfg.kernel_gt = '../dataset/DIV2K/kernel_{}.mat'.format(cfg.degradation_test)

        cfg.n_feats = 48
        cfg.depth = 4
        cfg.batch_size = 16
        if 'self' in template:
            cfg.loss = 'loss/ds_self.txt'
            cfg.trainer = 'downsampler.self'
            cfg.save = 'kernels_self'
            cfg.epochs = 20
            cfg.milestones = [5, 10, 15]
        else:
            cfg.loss = 'loss/ds_iterative.txt'
            cfg.trainer = 'downsampler.iterative'
            cfg.save = 'kernels_16'
            cfg.epochs = 30
            cfg.milestones = [10, 15, 20]

        if 'jaeha' in template:
            cfg.model = 'jaeha.generator'
            cfg.dis = 'jaeha.discriminator'
            cfg.loss = 'loss/ds_jaeha.txt'
            cfg.trainer = 'jaeha.unpaired'

            cfg.lr = 5e-5
            cfg.save = 'kernels_jaeha'
            cfg.epochs = 80
            cfg.milestones = [987654321]
        else:
            cfg.model = 'downsampler.dnew'
            cfg.dis = 'downsampler.discriminator_kgan'

        cfg.depth_sub = 7
        cfg.width_sub = 64
        cfg.adjust_weight = 0.010
        cfg.reset = True

    '''
    if 'ft' in template:
        if 'face' in template:
            cfg.scale = 8
            cfg.dtrain = ['sr.celeba_mask']
            cfg.dtest = ['sr.celeba_mask']
            cfg.n_classes = 10
        else:
            cfg.scale = 4
            cfg.dtrain = ['sr.mixed']
            cfg.dtest = ['sr.mixed']
            cfg.n_classes = 8

        if 'rrdb' in template:
            cfg.model = 'sr.rrdb'

        if not cfg.resume:
            if 'face' in template:
                cfg.resume = 'dl-edsr-baseline-face-x8'
            else:
                if 'rrdb' in template:
                    cfg.resume = 'dl-rrdb-x4'
                else:
                    cfg.resume = 'dl-edsr-baseline-x4'

        if 'mixed' in template:
            cfg.use_div2k = True
            cfg.use_ost = True
            cfg.use_flickr = False
        if 'div2k' in template:
            cfg.use_div2k = True
        if 'ost' in template:
            cfg.use_ost = True
        if 'df2k' in template:
            cfg.use_div2k = True
            cfg.use_flickr = True
        if 'all' in template:
            cfg.use_div2k = True
            cfg.use_flickr = True
            cfg.use_ost = True

        cfg.lr = 1e-4
        cfg.gan_k = 0
        if cfg.use_patch and cfg.use_div2k:
            if cfg.use_flickr:
                cfg.epochs = 28
                cfg.milestones = [4, 7, 14, 21]
            else:
                if cfg.use_ost:
                    cfg.epochs = 95
                    cfg.milestones = [12, 24, 48, 72]
                else:
                    cfg.epochs = 112
                    cfg.milestones = [14, 28, 56, 84]
        else:
            cfg.epochs = 200
            cfg.milestones = [25, 50, 100, 150]
            if 'face' in template:
                cfg.epochs //= 2
                cfg.milestones = [d // 2 for d in cfg.milestones]

        if 'more' in template:
            cfg.epochs = int(1.5 * cfg.epochs)
            cfg.milestones = [int(1.5 * d) for d in cfg.milestones]

        if 'madv' in template:
            cfg.loss = 'loss/sr_mask.txt'
            cfg.trainer = 'sr.mask'
            if 'old' in template:
                cfg.dis = 'mask.discriminator_old'
            elif 'early' in template:
                cfg.dis = 'mask.discriminator'
                # Mask is applied at the end of classification layer,
                # so the scale doesn't change. Use early_stop to modify the model
                cfg.dis_early_fork = 1
                cfg.mask_scale = 16
                # Override
                if 'early1' in template:
                    cfg.dis_early_fork = 1
                    cfg.mask_scale = 16
                elif 'early2' in template:
                    cfg.dis_early_fork = 2
                    cfg.mask_scale = 16
            elif 'seg' in template:
                cfg.dis = 'mask.discriminator_seg'
                cfg.dis_seg_model = 'segmentation/model.pt'
                cfg.dis_seg_n_feat = 32
                cfg.mask_scale = 16
                # Override
                if 'segd' in template:
                    cfg.dis = 'mask.discriminator_segdeep'
                # TODO type other segmentation network arguments here
            else:
                # Default
                cfg.dis = 'mask.discriminator'
        else:
            cfg.no_mask = True
            cfg.loss = 'loss/sr_adversarial.txt'
            cfg.dis = 'srgan.discriminator'
            cfg.dpatch = 0
    '''
