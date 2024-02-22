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
