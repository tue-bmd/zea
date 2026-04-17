import numpy as np
from keras import layers, metrics, ops, optimizers


class SoftPlus(layers.Layer):
    r"""
    Custom SoftPlus function that has a beta-parameter for gradient smoothing.

    .. math::
        f(x) = \frac{1}{\beta} \log(1 + \exp(\beta \cdot x))
    """

    def __init__(self, gradient_smoothing, min=-10):
        super().__init__(trainable=False)
        assert gradient_smoothing > 0, "gradient_smoothing must be greater than 0"
        self.gradient_smoothing = gradient_smoothing
        self.min = min

    def call(self, std):
        std = (1 / self.gradient_smoothing) * ops.logaddexp(0, self.gradient_smoothing * std)
        return ops.maximum(std, self.min)


class GaussianAnalyticalKL:
    def call(self, q_mu, q_std, p_mu, p_std):
        """
        KL-Divergence for two Gaussians q and p.
        """
        term1 = (q_mu - p_mu) * ops.reciprocal(p_std)
        term2 = q_std * ops.reciprocal(p_std)
        loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - ops.log(term2)
        return loss


class Parameters:
    """
    Class to hold all parameters for the HierarchicalVAE architecture and training.

    Args:
        args: Argument parser containing all necessary parameters. Loaded in from HuggingFace.
    """

    def __init__(self, args):
        super().__init__()

        self.save = args.save
        self.save_dir = args.save_dir
        self.dataset = args.dataset
        self.batch_size_div = max(1, args.batch_size_div)
        self.batch_size = int(args.batch_size // self.batch_size_div)

        if hasattr(args, "retrain_encoder"):
            self.retrain_encoder = args.retrain_encoder
        else:
            self.retrain_encoder = False
        if hasattr(args, "num_lines"):
            self.num_lines = args.num_lines
        else:
            self.num_lines = 256

        self.jit = "auto" if args.jit else False
        self.gpu = args.gpu

        # Model parameters
        self.b_act_name = args.b_act
        self.p_act_name = args.p_act
        self.block_activation = self.get_activation(args.b_act)
        self.pool_activation = self.get_activation(args.p_act)

        self.init_zeros = args.init_zeros

        self.use_spatial_attention = args.use_spatial_attention
        self.use_depthwise_attention = args.use_depthwise_attention
        self.query_width = args.query_width
        self.num_queries = args.num_queries

        self.block_bn = args.block_gn
        self.depthwise = args.depthwise
        self.num_output_mixtures = args.num_output_mixtures
        self.gradient_smoothing = args.gradient_smoothing
        self.gradient_clipnorm = args.gradient_clipnorm if args.gradient_clipnorm > 0 else None
        self.gradient_skipnorm = (
            args.gradient_skipnorm if args.gradient_skipnorm > 0 else 1e9
        )  # If it is 0, we make the threshold very large to effectively turn it off

        self.flow_type = args.flow_type
        self.spectral_norm = args.spectral_norm
        self.add_dataset_context()  # Writes new params
        self.set_stage_parameters(args)  # Writes new params

        # Number of stochastic layers
        self.model_depth = sum(self.dec_num_blocks)

        # Hyper parameters
        self.epochs = int(args.epochs)

        self.early_stopping = int(args.early_stopping)
        self.beta = float(args.beta)
        self.beta_warmup_epochs = int(args.beta_warmup_epochs)
        self.cyclic_beta = bool(args.cyclic_beta)
        self.num_cycles = int(args.number_cycles)

        self.learning_rate = float(args.learning_rate)
        self.learning_rate_end = float(args.learning_rate_end)
        self.lr_warmup_epochs = int(args.lr_warmup_epochs)

        self.weight_decay = float(args.weight_decay)
        self.use_ema = bool(args.use_ema)

        self.optimizer = str(args.optimizer)
        self.scheduler = str(args.scheduler)

        self.get_kernelsizes(args.increase_kernelsize)

        self.verify_parameters()  # Does not write new params

    def get_activation(self, name):
        return layers.Activation(name)

    def add_dataset_context(self):
        if self.dataset == "cifar10":
            self.enc_input_size = [32, 16, 8, 4, 2, 1]
            self.data_width = 3
            self.n_bits = 8
            self.loss_fn = DiscMixLogistic(
                self.n_bits,
                self.num_output_mixtures,
                self.data_width,
            )
            self.channels_out = (self.data_width * 3 + 1) * self.num_output_mixtures
            train_len = 50000

        elif self.dataset == "echonet":
            self.enc_input_size = [128, 64, 32, 16, 8, 4, 2, 1]
            self.data_width = 1
            self.n_bits = 8
            self.loss_fn = DiscMixLogistic(
                self.n_bits,
                self.num_output_mixtures,
                self.data_width,
            )
            self.channels_out = (self.data_width * 3 + 1) * self.num_output_mixtures
            train_len = 1222276  # 1222276

        elif self.dataset == "imagenet32":
            self.enc_input_size = [32, 16, 8, 4, 2, 1]
            self.data_width = 3
            self.n_bits = 8
            self.loss_fn = DiscMixLogistic(
                self.n_bits,
                self.num_output_mixtures,
                self.data_width,
            )
            self.channels_out = (self.data_width * 3 + 1) * self.num_output_mixtures
            train_len = 1281167  # 1281167

        elif self.dataset == "celeba64":
            self.enc_input_size = [64, 32, 16, 8, 4, 2, 1]
            self.data_width = 3
            self.n_bits = 8
            self.loss_fn = DiscMixLogistic(
                self.n_bits,
                self.num_output_mixtures,
                self.data_width,
            )
            self.channels_out = (self.data_width * 3 + 1) * self.num_output_mixtures
            train_len = 162770

        elif self.dataset == "echonetlvh":
            self.enc_input_size = [256, 128, 64, 32, 16, 8, 4, 2, 1]
            self.data_width = 3
            self.n_bits = 8
            self.loss_fn = DiscMixLogistic(
                self.n_bits,
                self.num_output_mixtures,
                self.data_width,
            )
            self.channels_out = (self.data_width * 3 + 1) * self.num_output_mixtures
            train_len = 1674377

        else:
            raise ValueError("No valid dataset was selected")

        self.step_per_epoch = int(
            np.floor(train_len / self.batch_size)
        )  # We floor since drop_remainder=True in the dataloaders

        self.dec_input_size = list(reversed(self.enc_input_size))
        self.num_stages = len(self.enc_input_size)

    def set_stage_parameters(self, args):
        # Repeats the parameters to the number of stages, if only a single input is given.
        # Reverses the decoder params to go from top-down to bottom-up.
        # Determines the pool widths for the convs inbetween stages.
        self.zdim = (
            list(reversed(args.z_width))
            if (len(args.z_width) == self.num_stages)
            else args.z_width * self.num_stages
        )
        self.num_flows = (
            list(reversed(args.num_flows))
            if (len(args.num_flows) == self.num_stages)
            else args.num_flows * self.num_stages
        )
        self.flow_in_ch = (
            list(reversed(args.flow_in_ch))
            if (len(args.flow_in_ch) == self.num_stages)
            else args.flow_in_ch * self.num_stages
        )
        self.num_ortho_vecs = (
            list(reversed(args.num_ortho_vecs))
            if (len(args.num_ortho_vecs) == self.num_stages)
            else args.num_ortho_vecs * self.num_stages
        )

        self.convsylv_channels = (
            list(reversed(args.convsylv_channels))
            if (len(args.convsylv_channels) == self.num_stages)
            else args.convsylv_channels * self.num_stages
        )
        self.convsylv_flows_per_stage = (
            list(reversed(args.convsylv_flows_per_stage))
            if (len(args.convsylv_flows_per_stage) == self.num_stages)
            else args.convsylv_flows_per_stage * self.num_stages
        )

        self.convsylv_splitfirst = (
            list(reversed(args.convsylv_splitfirst))
            if (len(args.convsylv_splitfirst) == self.num_stages)
            else args.convsylv_splitfirst * self.num_stages
        )

        self.convsylv_stage_limit = (
            list(reversed(args.convsylv_stage_limit))
            if (len(args.convsylv_stage_limit) == self.num_stages)
            else args.convsylv_stage_limit * self.num_stages
        )

        self.enc_in_width = (
            args.stage_in_width
            if (len(args.stage_in_width) == self.num_stages)
            else args.stage_in_width * self.num_stages
        )
        self.enc_middle_width = (
            args.enc_middle_width
            if (len(args.enc_middle_width) == self.num_stages)
            else args.enc_middle_width * self.num_stages
        )
        self.enc_num_blocks = (
            args.enc_num_blocks
            if (len(args.enc_num_blocks) == self.num_stages)
            else args.enc_num_blocks * self.num_stages
        )
        self.enc_pool_width = np.roll(self.enc_in_width, -1)
        self.enc_pool_width[-1] = self.enc_pool_width[-2]

        self.enc_sa_width = (
            args.s_a_width
            if (len(args.s_a_width) == self.num_stages)
            else args.s_a_width * self.num_stages
        )
        self.dec_sa_width = list(reversed(self.enc_sa_width))

        self.dec_in_width = list(reversed(self.enc_in_width))
        self.dec_middle_width = (
            list(reversed(args.dec_middle_width))
            if (len(args.dec_middle_width) == self.num_stages)
            else args.dec_middle_width * self.num_stages
        )
        self.dec_num_blocks = (
            list(reversed(args.dec_num_blocks))
            if (len(args.dec_num_blocks) == self.num_stages)
            else args.dec_num_blocks * self.num_stages
        )
        self.dec_pool_width = np.roll(self.dec_in_width, -1)
        self.dec_pool_width[-1] = self.dec_pool_width[-2]
        self.output_blocks = args.output_blocks

        self.z_out = (
            self.dec_input_size[-1],
            self.dec_input_size[-1],
            self.dec_in_width[-1],
        )
        self.z_out_width = args.z_out_width
        self.z_out_middle_width = args.z_out_middle_width

    def get_optimizer(self):
        if self.scheduler == "none":
            lr_scheduler = self.learning_rate
        elif self.scheduler == "exp":
            lr_scheduler = optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=self.step_per_epoch,
                decay_rate=0.977,  # 100 epochs for 10x decrease
            )
        elif self.scheduler == "cosd":
            w_steps = self.lr_warmup_epochs * self.step_per_epoch
            lr_scheduler = optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-6,
                decay_steps=self.step_per_epoch * self.epochs,
                alpha=self.learning_rate_end,
                warmup_target=self.learning_rate,
                warmup_steps=w_steps,
            )
        elif self.scheduler == "cosdr":
            lr_scheduler = optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=self.learning_rate,
                first_decay_steps=self.step_per_epoch * 2,
                t_mul=2.0,
                m_mul=1.0,
                alpha=0.0,
            )
        else:
            raise ValueError("invalid lr scheduler")

        grad_acc = self.batch_size_div if self.batch_size_div > 1 else None
        if self.optimizer == "adamax":
            opt = optimizers.Adamax(
                learning_rate=lr_scheduler,
                epsilon=1e-7,
                beta_1=0.9,
                beta_2=0.999,
                global_clipnorm=self.gradient_clipnorm,
                use_ema=self.use_ema,
                ema_momentum=0.99,
                weight_decay=self.weight_decay,
                gradient_accumulation_steps=grad_acc,
            )
        elif self.optimizer == "adamw":
            opt = optimizers.AdamW(
                learning_rate=lr_scheduler,
                global_clipnorm=self.gradient_clipnorm,
                use_ema=self.use_ema,
                ema_momentum=0.9999,
                weight_decay=self.weight_decay,
                gradient_accumulation_steps=grad_acc,
            )
        elif self.optimizer == "sgd":
            opt = optimizers.SGD(
                learning_rate=lr_scheduler,
                global_clipnorm=self.gradient_clipnorm,
                use_ema=self.use_ema,
                ema_momentum=0.9999,
                weight_decay=self.weight_decay,
                gradient_accumulation_steps=grad_acc,
            )
        else:
            raise ValueError("invalid optimizer")
        return opt

    def get_kernelsizes(self, increase_kernelsize):
        if increase_kernelsize:
            # Growing kernelsizes with resolution
            self.kernelsizes = {
                "1024": 13,
                "512": 11,
                "256": 7,
                "128": 5,
                "64": 3,
                "32": 3,
                "16": 3,
                "8": 3,
                "4": 3,
                "2": 1,
                "1": 1,
            }
        else:
            # kernelsize is always 3 or 1
            self.kernelsizes = {
                "1024": 3,
                "512": 3,
                "256": 3,
                "128": 3,
                "64": 3,
                "32": 3,
                "16": 3,
                "8": 3,
                "4": 3,
                "2": 1,
                "1": 1,
            }

    def verify_parameters(self):
        # All these params should have one value per stage
        assert len(self.zdim) == self.num_stages
        assert len(self.enc_in_width) == self.num_stages
        assert len(self.enc_middle_width) == self.num_stages
        assert len(self.enc_num_blocks) == self.num_stages

        assert len(self.dec_in_width) == self.num_stages
        assert len(self.dec_middle_width) == self.num_stages
        assert len(self.dec_num_blocks) == self.num_stages

        assert self.beta_warmup_epochs == int(self.beta_warmup_epochs)
        assert self.num_output_mixtures == int(self.num_output_mixtures)

        if self.cyclic_beta:
            assert self.num_cycles * self.beta_warmup_epochs < self.epochs

        # No negative warmup epochs
        assert self.beta_warmup_epochs >= 0

        # If we use depthwise attention, make sure all stages have at least one block
        # if self.use_depthwise_attention:
        #     assert all(x > 0 for x in self.enc_num_blocks)
        #     assert all(x > 0 for x in self.dec_num_blocks)

        # integers in attention width
        assert all(isinstance(x, int) for x in self.enc_sa_width)
        assert all(isinstance(x, int) for x in self.dec_sa_width)

        for i in range(len(self.enc_input_size) - 1):
            # Make sure every input_size is a decreasing power of 2.
            assert self.enc_input_size[i] // 2 == self.enc_input_size[i + 1]
            assert ops.log2(self.enc_input_size[i]) == int(ops.log2(self.enc_input_size[i]))

        # If we are using an orthogonal sylvester flow in a stage,
        # it cannot have more orthogonal vectors than z_dimensions
        if self.flow_type != "none":
            for vec, z, num, input_shape in zip(
                self.num_ortho_vecs, self.zdim, self.num_flows, self.enc_input_size
            ):
                if num > 0:
                    assert vec <= z * (input_shape**2)


class DiscMixLogistic:
    """
    Discretized Mixture of Logistics loss function.
    """

    def __init__(self, num_bits, num_mixtures, num_channels):
        """
        Args:
            num_bits (int): Number of bits used for pixel representation.
            num_mixtures (int): Number of mixture components in the model.
            num_channels (int): Number of channels in the input images.
        """
        self.reduction = "none"
        self.num_bits = num_bits
        self.num_mixtures = num_mixtures
        self.num_channels = num_channels
        self.num_classes = 2.0**self.num_bits - 1.0
        self.min_pix_value = -1
        self.max_pix_value = 1
        self.min_mol_logscale = -250
        self.softplus = SoftPlus(0.69314718056)

        self.cone_loss_mask = cone_loss_mask(
            batch_size=1, r_max=256, angle=0.7854, shape=(256, 256)
        )
        assert ops.shape(self.cone_loss_mask) == (1, 256, 256)
        self.dt = "float32"

    def call(self, targets, logits, mask=None):
        """
        Calculates the negative log-likelihood of the targets given the model logits.

        Args:
            targets (tensor): Ground truth images of shape [B, H, W, C].
            logits (tensor): Model output logits of shape [B, H, W, M * (3 * C + 1)].
            mask (tensor, optional): Binary mask of shape [B, H, W, C] to apply to the loss.

        Returns:
            loss (tensor): Negative log-likelihood loss of shape [B].
        """

        B, H, W, C = (
            ops.shape(targets)[0],
            ops.shape(targets)[1],
            ops.shape(targets)[2],
            ops.shape(targets)[3],
        )
        assert C == 3 or C == 1  # Only RGB or grayscale images are supported

        targets = ops.cast(ops.expand_dims(targets, -1), dtype=self.dt)  # B, H, W, C, 1

        logit_probs = logits[:, :, :, : self.num_mixtures]  # B, H, W, M * 1
        lg = logits[:, :, :, self.num_mixtures :]  # B, H, W, M*C*3
        lg = layers.Reshape([H, W, self.num_channels, 3 * self.num_mixtures])(lg)  # B, H, W, C, 3*M

        model_means = lg[:, :, :, :, : self.num_mixtures]  # B, H, W, C, M

        log_scales = self.min_mol_logscale + self.softplus(
            lg[:, :, :, :, self.num_mixtures : 2 * self.num_mixtures] - self.min_mol_logscale
        )

        model_coeffs = ops.tanh(
            lg[:, :, :, :, 2 * self.num_mixtures : 3 * self.num_mixtures]
        )  # B, H, W, C, M

        # RGB AR
        if C == 3:
            mean1 = model_means[:, :, :, 0:1, :]  # B, H, W, 1, M
            mean2 = ops.add(
                model_means[:, :, :, 1:2, :],
                ops.multiply(model_coeffs[:, :, :, 0:1, :], targets[:, :, :, 0:1, :]),
            )  # B, H, W, 1, M
            mean3 = ops.add(
                ops.add(
                    model_means[:, :, :, 2:3, :],
                    ops.multiply(model_coeffs[:, :, :, 1:2, :], targets[:, :, :, 0:1, :]),
                ),
                ops.multiply(model_coeffs[:, :, :, 2:3, :], targets[:, :, :, 1:2, :]),
            )  # B, H, W, 1, M
            means = ops.concatenate([mean1, mean2, mean3], axis=3)  # B, H, W, C, M
        else:
            means = model_means
        centered = targets - means  # B, H, W, C, M

        inv_stdv = ops.exp(-log_scales)  # B, H, W, C, M
        plus_in = ops.multiply(inv_stdv, (centered + 1.0 / self.num_classes))
        cdf_plus = ops.sigmoid(plus_in)
        min_in = ops.multiply(inv_stdv, (centered - 1.0 / self.num_classes))
        cdf_min = ops.sigmoid(min_in)

        log_cdf_plus = plus_in - ops.softplus(
            plus_in
        )  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -ops.softplus(
            min_in
        )  # log probability for edge case of 255 (before scaling)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min  # B, H, W, C, M

        mid_in = ops.multiply(inv_stdv, centered)
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in this code)
        log_pdf_mid = mid_in - log_scales - 2.0 * ops.softplus(mid_in)

        # the original implementation uses samples > 0.999,
        # this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        broadcast_targets = ops.broadcast_to(targets, shape=(B, H, W, C, self.num_mixtures))

        # Explanation of the nested where statements:
        # First where statement, choose log probability of 0 if target == 0, else continue to:
        # Second where statement, choose log probability of 255 if target == 255 else continue to:
        # Third where statement:
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py line 77
        log_probs = ops.where(
            broadcast_targets < (self.min_pix_value + 0.001),
            log_cdf_plus,
            ops.where(
                broadcast_targets > (self.max_pix_value - 0.001),
                log_one_minus_cdf_min,
                ops.where(
                    cdf_delta > 1e-5,
                    ops.log(ops.maximum(cdf_delta, 1e-12)),
                    ops.subtract(log_pdf_mid, ops.log(self.num_classes / 2)),
                ),
            ),
        )  # B, H, W, C, M

        # ----- NEW: collapse mixtures per-channel, then apply mask per-channel -----
        # log mixture weights: [B,H,W,M]
        log_pi = ops.log_softmax(logit_probs, axis=3)  # axis=3 as in your original code
        log_pi_exp = ops.expand_dims(log_pi, axis=3)  # [B,H,W,1,M] to broadcast over channels
        # log p(x_c) = logsumexp_m [ log p(x_c | m) + log pi_m ]  -> [B,H,W,C]
        logp_per_channel = ops.logsumexp(log_probs + log_pi_exp, axis=-1)  # collapse mixtures
        neglogp_per_channel = -logp_per_channel  # [B,H,W,C]
        # apply user mask per-channel
        if mask is not None:
            mask = ops.cast(mask, dtype="float32")
            masked_neglog = neglogp_per_channel * mask  # [B,H,W,C]
            # Sum over channel dimension
            neg_log_probs = ops.sum(masked_neglog, axis=3)  # [B, H, W]
        else:
            neg_log_probs = ops.sum(neglogp_per_channel, axis=3)  # [B, H, W]
        neg_log_probs *= self.cone_loss_mask  # Apply cone loss mask
        loss = ops.sum(neg_log_probs, axis=[1, 2])
        if mask is not None:
            # normalize by number of unmasked pixels
            num_pixels = ops.sum(mask, axis=[1, 2, 3])
            scale = ops.clip(
                256 * 256 / num_pixels, 1.0, 256
            )  # scale is 1 for fully sampled, 256 for 1 line.
            loss *= scale
        return loss


def cone_loss_mask(batch_size, r_max=107, angle=0.79, shape=(112, 112)):
    """
    Cone loss mask for scan-converted reconstruction.
    Models are trained in polar domain,
    this mask weights the loss as if the image was in cartesian domain.
    """
    sector = 2 * angle / (2 * 3.1415)
    radii_rows = ops.linspace(0.1, r_max, shape[0])
    distance_of_rows = sector * 2 * 3.1415 * radii_rows
    distance_of_rows /= ops.mean(distance_of_rows)
    weights = ops.expand_dims(ops.expand_dims(distance_of_rows, 0), -1)
    weights = ops.repeat(weights, shape[1], axis=2)
    return weights


class GradientNorms(metrics.Metric):
    """
    Metric to track gradient norms during training.
    """

    def __init__(self, name="grad_norm", **kwargs):
        super(GradientNorms, self).__init__(name=name, **kwargs)
        self.grads = self.add_variable(
            shape=(),
            initializer="zeros",
            name="grads",
        )

    def update_state(self, grads, sample_weight=None):
        grads = ops.cast(grads, dtype="float32")
        self.grads.assign(grads)

    def result(self):
        return self.grads
