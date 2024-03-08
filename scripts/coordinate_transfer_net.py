import torch


class ImplicitCoordinateTransfer:
    def __init__(self, model_weight_path, device):
        self.device = device
        self.model = model = torch.nn.Sequential(
            torch.nn.Linear(4, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 1),
        )
        model.load_state_dict(torch.load(model_weight_path))
        model.to(device)
        model.eval()

    def create_uniform_samples(self, observation, n):
        return torch.cat(
            (
                observation.unsqueeze(-2).repeat_interleave(n, dim=-2),
                torch.distributions.Uniform(-1.1, 1.1).sample(
                    (observation.shape[0], n, 2)
                ),
            ),
            -1,
        ).to(self.device)

    def derivative_free_optimizer(
        self,
        observation,
        n_samples=16384,
        n_iters=4,
        initial_sigma=0.01,
        sigma_decay=0.1,
    ):
        with torch.no_grad():
            samples = self.create_uniform_samples(observation, n_samples)
            sigma = torch.tensor(initial_sigma).to(self.device)
            for i in range(n_iters - 1):
                probs = self.model(samples).squeeze(-1).softmax(-1)
                samples_i = probs.multinomial(n_samples, replacement=True)
                samples = samples.index_select(-2, samples_i.squeeze())
                samples[:, :, 2:] += sigma.sqrt() * torch.rand(
                    samples[:, :, 2:].shape
                ).to(self.device)
                # FIXME clip to bounds ommitted to enable extrapolation
                sigma *= sigma_decay
            y_i = self.model(samples).squeeze(-1).softmax(-1).argmax(-1)
            return samples[torch.arange(samples.shape[0]), y_i, 2:]


class ImageCoordinateFilter:
    def __init__(
        self,
        workspace=torch.tensor(
            [
                [0.0396, 0.7160],
                [0.2021, 0.3444],
                [0.7646, 0.3278],
                [0.9448, 0.7313],
                [0.8162, 0.8069],
                [0.6391, 0.8632],
                [0.4380, 0.8757],
                [0.2599, 0.8375],
                [0.1328, 0.7771],
            ]
        ),
    ):
        self.workspace = workspace

    def is_within_workspace(self, point):
        cross_products = torch.tensor(
            [
                (point[0] - self.workspace[i - 1][0])
                * (self.workspace[i][1] - self.workspace[i - 1][1])
                - (self.workspace[i][0] - self.workspace[i - 1][0])
                * (point[1] - self.workspace[i - 1][1])
                for i in range(len(self.workspace))
            ],
        )
        return torch.logical_or(
            torch.all(cross_products <= 0), torch.all(cross_products >= 0)
        )
