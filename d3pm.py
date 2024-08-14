import os
import random
import sys

import numpy as np
import paddle
import paddle.nn as nn
from PIL import Image
from tqdm import tqdm

# This code is highly referenced from https://github.com/cloneofsimo/d3pm


def blk(ic, oc):
    return nn.Sequential(
        nn.Conv2D(in_channels=ic, out_channels=oc, kernel_size=5, padding=2),
        nn.GroupNorm(num_groups=oc // 8, num_channels=oc),
        nn.LeakyReLU(),
        nn.Conv2D(in_channels=oc, out_channels=oc, kernel_size=5, padding=2),
        nn.GroupNorm(num_groups=oc // 8, num_channels=oc),
        nn.LeakyReLU(),
        nn.Conv2D(in_channels=oc, out_channels=oc, kernel_size=5, padding=2),
        nn.GroupNorm(num_groups=oc // 8, num_channels=oc),
        nn.LeakyReLU(),
    )


def blku(ic, oc):
    return nn.Sequential(
        nn.Conv2D(in_channels=ic, out_channels=oc, kernel_size=5, padding=2),
        nn.GroupNorm(num_groups=oc // 8, num_channels=oc),
        nn.LeakyReLU(),
        nn.Conv2D(in_channels=oc, out_channels=oc, kernel_size=5, padding=2),
        nn.GroupNorm(num_groups=oc // 8, num_channels=oc),
        nn.LeakyReLU(),
        nn.Conv2D(in_channels=oc, out_channels=oc, kernel_size=5, padding=2),
        nn.GroupNorm(num_groups=oc // 8, num_channels=oc),
        nn.LeakyReLU(),
        nn.Conv2DTranspose(in_channels=oc, out_channels=oc, kernel_size=2, stride=2),
        nn.GroupNorm(num_groups=oc // 8, num_channels=oc),
        nn.LeakyReLU(),
    )


class DummyX0Model(nn.Layer):
    def __init__(self, n_channel: int, N: int = 16) -> None:
        super().__init__()
        self.down1 = blk(n_channel, 16)
        self.down2 = blk(16, 32)
        self.down3 = blk(32, 64)
        self.down4 = blk(64, 512)
        self.down5 = blk(512, 512)
        self.up1 = blku(512, 512)
        self.up2 = blku(512 + 512, 64)
        self.up3 = blku(64, 32)
        self.up4 = blku(32, 16)
        self.convlast = blk(16, 16)
        self.final = nn.Conv2D(
            in_channels=16, out_channels=N * n_channel, kernel_size=1, bias_attr=False
        )
        self.tr1 = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048
        )
        self.tr2 = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048
        )
        self.tr3 = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=2048)
        self.cond_embedding_1 = nn.Embedding(num_embeddings=10, embedding_dim=16)
        self.cond_embedding_2 = nn.Embedding(num_embeddings=10, embedding_dim=32)
        self.cond_embedding_3 = nn.Embedding(num_embeddings=10, embedding_dim=64)
        self.cond_embedding_4 = nn.Embedding(num_embeddings=10, embedding_dim=512)
        self.cond_embedding_5 = nn.Embedding(num_embeddings=10, embedding_dim=512)
        self.cond_embedding_6 = nn.Embedding(num_embeddings=10, embedding_dim=64)
        self.temb_1 = nn.Linear(in_features=32, out_features=16)
        self.temb_2 = nn.Linear(in_features=32, out_features=32)
        self.temb_3 = nn.Linear(in_features=32, out_features=64)
        self.temb_4 = nn.Linear(in_features=32, out_features=512)
        self.N = N

    def forward(self, x, t, cond) -> paddle.Tensor:
        x = 2 * x.astype(dtype="float32") / self.N - 1.0
        t = t.astype(dtype="float32").reshape([-1, 1]) / 1000
        t_features = [paddle.sin(x=t * 3.1415 * 2**i) for i in range(16)] + [
            paddle.cos(x=t * 3.1415 * 2**i) for i in range(16)
        ]
        tx = paddle.concat(x=t_features, axis=1)
        t_emb_1 = self.temb_1(tx).unsqueeze(axis=-1).unsqueeze(axis=-1)
        t_emb_2 = self.temb_2(tx).unsqueeze(axis=-1).unsqueeze(axis=-1)
        t_emb_3 = self.temb_3(tx).unsqueeze(axis=-1).unsqueeze(axis=-1)
        t_emb_4 = self.temb_4(tx).unsqueeze(axis=-1).unsqueeze(axis=-1)
        cond_emb_1 = self.cond_embedding_1(cond).unsqueeze(axis=-1).unsqueeze(axis=-1)
        cond_emb_2 = self.cond_embedding_2(cond).unsqueeze(axis=-1).unsqueeze(axis=-1)
        cond_emb_3 = self.cond_embedding_3(cond).unsqueeze(axis=-1).unsqueeze(axis=-1)
        cond_emb_4 = self.cond_embedding_4(cond).unsqueeze(axis=-1).unsqueeze(axis=-1)
        cond_emb_5 = self.cond_embedding_5(cond).unsqueeze(axis=-1).unsqueeze(axis=-1)
        cond_emb_6 = self.cond_embedding_6(cond).unsqueeze(axis=-1).unsqueeze(axis=-1)
        x1 = self.down1(x) + t_emb_1 + cond_emb_1
        x2 = (
            self.down2(nn.functional.avg_pool2d(kernel_size=2, x=x1, exclusive=False))
            + t_emb_2
            + cond_emb_2
        )
        x3 = (
            self.down3(nn.functional.avg_pool2d(kernel_size=2, x=x2, exclusive=False))
            + t_emb_3
            + cond_emb_3
        )
        x4 = (
            self.down4(nn.functional.avg_pool2d(kernel_size=2, x=x3, exclusive=False))
            + t_emb_4
            + cond_emb_4
        )
        x5 = self.down5(nn.functional.avg_pool2d(kernel_size=2, x=x4, exclusive=False))

        x5 = (
            self.tr1(x5.reshape([x5.shape[0], x5.shape[1], -1]).transpose([0, 2, 1]))
            .transpose([0, 2, 1])
            .reshape(x5.shape)
        )

        y = self.up1(x5) + cond_emb_5

        y = (
            self.tr2(y.reshape([y.shape[0], y.shape[1], -1]).transpose([0, 2, 1]))
            .transpose([0, 2, 1])
            .reshape(y.shape)
        )

        y = self.up2(paddle.concat(x=[x4, y], axis=1)) + cond_emb_6

        y = (
            self.tr3(y.reshape([y.shape[0], y.shape[1], -1]).transpose([0, 2, 1]))
            .transpose([0, 2, 1])
            .reshape(y.shape)
        )

        y = self.up3(y)
        y = self.up4(y)
        y = self.convlast(y)
        y = self.final(y)
        y = y.reshape([y.shape[0], -1, self.N, *x.shape[2:]]).transpose([0, 1, 4, 3, 2])
        return y


class D3PM(nn.Layer):
    def __init__(
        self,
        x0_model: nn.Layer,
        n_T: int,
        num_classes: int = 10,
        forward_type="uniform",
        hybrid_loss_coeff=0.001,
    ) -> None:
        super().__init__()
        self.x0_model = x0_model
        self.n_T = n_T
        self.hybrid_loss_coeff = hybrid_loss_coeff

        self.eps = 1e-06
        self.num_classes = num_classes
        q_onestep_mats = []
        q_mats = []

        if forward_type == "uniform":
            steps = paddle.arange(dtype="float64", end=n_T + 1) / n_T
            alpha_bar = paddle.cos(x=(steps + 0.008) / 1.008 * 3.1415926 / 2)
            self.beta_t = paddle.minimum(
                x=1 - alpha_bar[1:] / alpha_bar[:-1],
                y=paddle.ones_like(x=alpha_bar[1:]) * 0.999,
            )
            for beta in self.beta_t:
                mat = paddle.ones(shape=[num_classes, num_classes]) * beta / num_classes
                mat.diagonal().fill_(
                    value=1 - (num_classes - 1) * beta.item() / num_classes
                )
                q_onestep_mats.append(mat)
        elif forward_type == "absorbing":
            self.beta_t = 1.0 / paddle.linspace(n_T, 1.0, n_T)
            for beta in self.beta_t:
                diag = paddle.full(shape=(self.num_classes,), fill_value=1.0 - beta)
                mat = paddle.diag(diag, offset=0)
                mat[:, self.num_classes // 2] += beta
                q_onestep_mats.append(mat)
        else:
            raise NotImplementedError(
                f'{forward_type} not implemented, use one of ["uniform","absorbing"]'
            )
        q_one_step_mats = paddle.stack(x=q_onestep_mats, axis=0)
        x = q_one_step_mats
        q_one_step_transposed = x.transpose([0, 2, 1])
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = paddle.stack(x=q_mats, axis=0)
        self.logit_type = "logit"
        self.register_buffer(name="q_one_step_transposed", tensor=q_one_step_transposed)
        self.register_buffer(name="q_mats", tensor=q_mats)
        assert tuple(self.q_mats.shape) == (self.n_T, num_classes, num_classes), tuple(
            self.q_mats.shape
        )

    def _at(self, a, t, x):
        t = t.reshape((t.shape[0], *([1] * (x.dim() - 1))))
        return a[t - 1, x, :]

    def q_posterior_logits(self, x_0, x_t, t):

        if x_0.dtype == paddle.int64 or x_0.dtype == paddle.int32:
            x_0_logits = paddle.log(
                x=nn.functional.one_hot(num_classes=self.num_classes, x=x_0).astype(
                    "int64"
                )
                + self.eps
            )
        else:
            x_0_logits = x_0.clone()
        assert tuple(x_0_logits.shape) == tuple(x_t.shape) + (self.num_classes,)

        fact1 = self._at(self.q_one_step_transposed, t, x_t)
        softmaxed = nn.functional.softmax(x=x_0_logits, axis=-1)
        index = t - 2
        index = paddle.where(condition=index < 0, x=index + self.n_T, y=index)
        qmats2 = self.q_mats[index].cast(softmaxed.dtype)

        fact2 = paddle.einsum("bijkc,bcd->bijkd", softmaxed, qmats2)
        out = paddle.log(x=fact1 + self.eps) + paddle.log(x=fact2 + self.eps)
        t_broadcast = t.reshape((t.shape[0], *([1] * x_t.dim())))
        bc = paddle.where(condition=t_broadcast == 1, x=x_0_logits, y=out)
        return bc

    def vb(self, dist1, dist2):
        dist1 = dist1.flatten(start_axis=0, stop_axis=-2)
        dist2 = dist2.flatten(start_axis=0, stop_axis=-2)
        out = nn.functional.softmax(x=dist1 + self.eps, axis=-1) * (
            nn.functional.log_softmax(dist1 + self.eps, axis=-1)
            - nn.functional.log_softmax(dist2 + self.eps, axis=-1)
        )
        return out.sum(axis=-1).mean()

    def q_sample(self, x_0, t, noise):
        logits = paddle.log(x=self._at(self.q_mats, t, x_0) + self.eps)
        noise = paddle.clip(x=noise, min=self.eps, max=1.0)
        gumbel_noise = -paddle.log(x=-paddle.log(x=noise))
        return paddle.argmax(x=logits + gumbel_noise, axis=-1)

    def model_predict(self, x_0, t, cond):
        predicted_x0_logits = self.x0_model(x_0, t, cond)
        return predicted_x0_logits

    def forward(self, x: paddle.Tensor, cond: paddle.Tensor = None) -> paddle.Tensor:
        """
        Makes forward diffusion x_t from x_0, and tries to guess x_0 value from x_t using x0_model.
        x is one-hot of dim (bs, ...), with int values of 0 to num_classes - 1
        """
        t = paddle.randint(low=1, high=self.n_T, shape=(x.shape[0],))
        x_t = self.q_sample(x, t, paddle.rand(shape=(*x.shape, self.num_classes)))
        assert tuple(x_t.shape) == tuple(x.shape), print(
            f"x_t.shape: {tuple(x_t.shape)}, x.shape: {tuple(x.shape)}"
        )
        predicted_x0_logits = self.model_predict(x_t, t, cond)
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t)
        vb_loss = self.vb(true_q_posterior_logits, pred_q_posterior_logits)
        predicted_x0_logits = predicted_x0_logits.flatten(start_axis=0, stop_axis=-2)
        x = x.flatten(start_axis=0, stop_axis=-1)
        ce_loss = nn.functional.cross_entropy(predicted_x0_logits, x)
        return vb_loss + self.hybrid_loss_coeff * ce_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def p_sample(self, x, t, cond, noise):
        predicted_x0_logits = self.model_predict(x, t, cond)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)
        noise = paddle.clip(x=noise, min=self.eps, max=1.0)
        not_first_step = (
            (t != 1).astype(dtype="float32").reshape((x.shape[0], *([1] * x.dim())))
        )
        gumbel_noise = -paddle.log(x=-paddle.log(x=noise))
        sample = paddle.argmax(
            x=pred_q_posterior_logits + gumbel_noise * not_first_step, axis=-1
        )
        return sample

    def sample(self, x, cond=None):
        for t in reversed(range(1, self.n_T)):
            t = paddle.to_tensor(data=[t] * x.shape[0])
            x = self.p_sample(
                x, t, cond, paddle.rand(shape=(*x.shape, self.num_classes))
            )
        return x

    def sample_with_image_sequence(self, x, cond=None, stride=10):
        steps = 0
        images = []
        for t in reversed(range(1, self.n_T)):
            t = paddle.to_tensor(data=[t] * x.shape[0])
            x = self.p_sample(
                x, t, cond, paddle.rand(shape=(*x.shape, self.num_classes))
            )
            steps += 1
            if steps % stride == 0:
                images.append(x)
        if steps % stride != 0:
            images.append(x)
        return images


def set_random_seed(seed: int):
    """Set numpy, random, paddle random_seed to given seed.

    Args:
        seed (int): Random seed.
    """
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    set_random_seed(42)

    N = 2
    d3pm = D3PM(DummyX0Model(1, N), 1000, num_classes=N, hybrid_loss_coeff=1.0)

    dataset = paddle.vision.datasets.MNIST(
        mode="train",
        download=True,
        transform=paddle.vision.transforms.Compose(
            [paddle.vision.transforms.ToTensor(), paddle.vision.transforms.Pad(2)]
        ),
    )

    dataloader = paddle.io.DataLoader(
        dataset=dataset, batch_size=256, shuffle=True, num_workers=4
    )
    optim = paddle.optimizer.AdamW(
        parameters=d3pm.x0_model.parameters(), learning_rate=0.001, weight_decay=0.0
    )
    d3pm.train()
    n_epoch = 400
    global_step = 0
    for epoch in range(n_epoch):
        pbar = tqdm(dataloader)
        loss_ema = None
        for x, cond in pbar:
            cond = cond.squeeze()
            optim.clear_grad()
            x = (x * (N - 1)).round().astype(dtype="int64").clip(min=0, max=N - 1)
            loss, info = d3pm(x, cond)
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()
            pbar.set_description(
                f"Epoch: [{epoch}/{n_epoch}], loss: {loss_ema:.4f}, vb_loss: {info['vb_loss']:.4f}, ce_loss: {info['ce_loss']:.4f}"
            )
            optim.step()
        if (epoch + 1) % 10 == 0:
            os.makedirs("results", exist_ok=True)
            d3pm.eval()
            with paddle.no_grad():
                cond = paddle.arange(start=0, end=4) % 10
                init_noise = paddle.randint(low=0, high=N, shape=(4, 1, 32, 32))
                images = d3pm.sample_with_image_sequence(init_noise, cond, stride=40)
                gif = []
                for image in images:
                    image_shape = image.shape
                    x_as_image = image.astype(dtype="float32") / (N - 1)
                    x_as_image = (
                        x_as_image.squeeze()
                        .transpose([1, 0, 2])
                        .reshape([image_shape[2], image_shape[0] * image_shape[3]])
                    )
                    img = x_as_image.cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    gif.append(Image.fromarray(img))
                gif[0].save(
                    f"results/sample_{epoch}.gif",
                    save_all=True,
                    append_images=gif[1:],
                    duration=100,
                    loop=0,
                )
                last_img = gif[-1]
                last_img.save(f"results/sample_{epoch}_last.png")
            d3pm.train()
