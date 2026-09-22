"""CNN-LSTM challenger: reads the past week as a sequence and predicts the whole next-48-hour path at once."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..anomaly_detection.config import SENSORS
from .features import AIR, CLOUD, RAIN, forecast_error, row_info, take_rows


def _take_grid(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return take_rows(x, idx.ravel()).reshape(*idx.shape, x.shape[1])


def _forward_fill(a: np.ndarray) -> np.ndarray:
    """Fill gaps along the time axis of (samples, time, channels) with the last known value; leading gaps become 0."""
    steps = np.arange(a.shape[1])[None, :, None]
    last_known = np.maximum.accumulate(np.where(np.isnan(a), 0, steps), axis=1)
    return np.nan_to_num(np.take_along_axis(a, last_known, axis=1))


def _hour_features(tl, idx: np.ndarray) -> np.ndarray:
    angle = 2 * np.pi * tl.hour_of(idx) / 24
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1)


def _pinball(pred, target, quantiles):
    known = ~torch.isnan(target)
    err = torch.nan_to_num(target).unsqueeze(-1) - pred
    loss = torch.maximum(quantiles * err, (quantiles - 1) * err) * known.unsqueeze(-1)
    return loss.sum() / (known.sum() * len(quantiles)).clamp(min=1)


class _Net(nn.Module):
    def __init__(self, past_channels, future_channels, horizon, n_targets, n_quantiles, hidden):
        super().__init__()
        self.out_shape = (horizon, n_targets, n_quantiles)
        # Convolutions pick up local shape (the daily rise and fall); the LSTM carries it across the week.
        self.conv = nn.Sequential(
            nn.Conv1d(past_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(32, hidden, batch_first=True)
        self.future = nn.Sequential(nn.Flatten(), nn.Linear(horizon * future_channels, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, 128), nn.ReLU(), nn.Linear(128, horizon * n_targets * n_quantiles))

    def forward(self, past, future):
        encoded = self.conv(past.transpose(1, 2)).transpose(1, 2)
        _, (state, _) = self.lstm(encoded)
        return self.head(torch.cat([state[-1], self.future(future)], dim=1)).view(-1, *self.out_shape)


class CNNLSTMHourlyModel:
    kind = "cnn_lstm"

    def __init__(self, settings: dict, seed: int):
        self.settings = settings
        self.seed = seed

    def _scale_weather(self, w: np.ndarray) -> np.ndarray:
        out = w.copy()
        out[..., AIR] = (w[..., AIR] - self.air_mean_) / self.air_std_
        out[..., RAIN] = np.log1p(np.clip(w[..., RAIN], 0, None))
        return out

    def _arrays(self, tl, code, origins, rng=None):
        """Past window, future known inputs, and the target path (change from now, standardised)."""
        lookback = self.settings["challenger"]["lookback_hours"]
        horizon = max(self.settings["horizons"])
        origins = np.asarray(origins, dtype=int)
        past_idx = origins[:, None] + np.arange(-lookback + 1, 1)[None, :]
        future_idx = origins[:, None] + np.arange(1, horizon + 1)[None, :]

        past = [_take_grid((tl.x - self.mean_) / self.std_, past_idx), _hour_features(tl, past_idx)]
        future = [_hour_features(tl, future_idx)]
        if self.uses_weather_:
            past.append(self._scale_weather(_take_grid(tl.w, past_idx)))
            ahead = _take_grid(tl.w, future_idx)
            if rng is not None:
                lead = np.arange(1, horizon + 1)[None, :]
                for col, kind in ((AIR, "air"), (CLOUD, "cloud"), (RAIN, "rain")):
                    ahead[..., col] = forecast_error(kind, ahead[..., col], lead, rng)
            future.append(self._scale_weather(ahead))
        station = np.zeros((len(origins), lookback, self.n_stations_))
        if code is not None and not np.isnan(code):
            station[:, :, int(code)] = 1.0
        past.append(station)

        now = take_rows(tl.x, origins)
        path = _take_grid(tl.x, np.where(future_idx < tl.n, future_idx, -1))
        target = (path - now[:, None, :]) / self.std_
        return (
            _forward_fill(np.concatenate(past, axis=2)).astype(np.float32),
            np.nan_to_num(np.concatenate(future, axis=2)).astype(np.float32),
            target.astype(np.float32),
        )

    def _batches(self, timelines, codes, start_of, stop_of, rng):
        c = self.settings["challenger"]
        lookback, horizon = c["lookback_hours"], max(self.settings["horizons"])
        parts = []
        for station, tl in timelines.items():
            origins = np.arange(max(lookback, start_of(tl)), min(tl.n, stop_of(tl)) - horizon, c["origin_every_hours"])
            if len(origins):
                parts.append(self._arrays(tl, codes[station], origins, rng))
        return [torch.from_numpy(np.concatenate(p)) for p in zip(*parts)]

    def fit(self, timelines: dict, codes: dict, until, rng, validation=None) -> "CNNLSTMHourlyModel":
        c = self.settings["challenger"]
        torch.manual_seed(self.seed)
        self.n_stations_ = len(codes)
        self.uses_weather_ = next(iter(timelines.values())).w is not None
        history = np.vstack([tl.x[: max(0, min(tl.n, tl.position(until)))] for tl in timelines.values()])
        self.mean_, self.std_ = np.nanmean(history, axis=0), np.nanstd(history, axis=0) + 1e-6
        if self.uses_weather_:
            air = np.concatenate([tl.w[: max(0, tl.position(until)), AIR] for tl in timelines.values()])
            self.air_mean_, self.air_std_ = np.nanmean(air), np.nanstd(air) + 1e-6

        past, future, target = self._batches(timelines, codes, lambda tl: 0, lambda tl: tl.position(until), rng)
        check = None
        if validation is not None:
            start, end = validation
            check = self._batches(timelines, codes, lambda tl: tl.position(start), lambda tl: tl.position(end), rng)

        quantiles = torch.tensor(self.settings["quantiles"], dtype=torch.float32)
        self.net = _Net(past.shape[2], future.shape[2], future.shape[1], len(SENSORS), len(quantiles), c["hidden"])
        optimiser = torch.optim.Adam(self.net.parameters(), lr=c["learning_rate"])
        order = torch.Generator().manual_seed(self.seed)
        best, best_state, stale = np.inf, None, 0
        self.epochs_run_ = c["epochs"]
        for epoch in range(c["epochs"]):
            self.net.train()
            for batch in torch.randperm(len(past), generator=order).split(c["batch_size"]):
                optimiser.zero_grad()
                loss = _pinball(self.net(past[batch], future[batch]), target[batch], quantiles)
                loss.backward()
                optimiser.step()
            if check is None:
                continue
            self.net.eval()
            with torch.no_grad():
                score = float(_pinball(self.net(check[0], check[1]), check[2], quantiles))
            if score < best - 1e-4:
                best, best_state, stale = score, {k: v.clone() for k, v in self.net.state_dict().items()}, 0
                self.epochs_run_ = epoch + 1
            else:
                stale += 1
                if stale >= c["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_deltas(self, timelines: dict, codes: dict, origins: dict, rng=None):
        info_parts, preds = [], []
        for station, positions in origins.items():
            tl = timelines[station]
            past, future, _ = self._arrays(tl, codes.get(station), positions, rng)
            with torch.no_grad():
                out = self.net(torch.from_numpy(past), torch.from_numpy(future)).numpy()
            out = np.sort(out, axis=-1) * self.std_[None, None, :, None]
            horizons = np.arange(1, out.shape[1] + 1)
            info_parts.append(row_info(tl, np.repeat(positions, len(horizons)), np.tile(horizons, len(positions))))
            preds.append(out.reshape(-1, len(SENSORS), out.shape[-1]))
        pred = np.concatenate(preds)
        return pd.concat(info_parts, ignore_index=True), {s: pred[:, k, :] for k, s in enumerate(SENSORS)}
