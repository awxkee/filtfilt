# filtfilt

A Rust implementation of zero-phase forward-backward IIR filtering.

Useful in signal processing, biosignal analysis, and time-series smoothing where phase distortion must be avoided.

It supports odd and even edge-extension padding to suppress boundary transients, and operates internally in `f64` for numerical stability even when filtering `f32` signals.

## Features

- Zero-phase filtering via forward + backward pass
- Odd (anti-symmetric) and even (symmetric) edge extension
- `f32` and `f64` input support
- Numerically stable — uses initial conditions (`zi`) to settle filter state before each pass

## Usage
```rust
use filtfilt::{filtfilt, FilterOptions, FilterPadding};
use std::borrow::Cow;

let signal: Vec = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();

let y = filtfilt(&signal, FilterOptions {
    b: Cow::Borrowed(&[0.2, 0.5, 0.2]),
    a: Cow::Borrowed(&[1.0, -0.3, 0.05]),
    pad_type: FilterPadding::Odd,
});
```

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
