# Bevy Radix Sort

A **low-level**, high-performance GPU-based radix sort plugin for Bevy, optimized for sorting key/val of type `u32`.

[![Crates.io](https://img.shields.io/crates/v/bevy_radix_sort.svg)](https://crates.io/crates/bevy_radix_sort)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/AllenPocketGamer/bevy_radix_sort#license)

## Features

- Based on the paper [Fast 4-way parallel radix sorting on GPUs](http://www.sci.utah.edu/~csilva/papers/cgf.pdf)
- High-performance `radix sort` implementation fully executed on the GPU
- High compatibility, capable of running on most GPUs

## Limitations

- Currently not supported on web platforms(due to the lack of push_constants support in the WebGPU standard)

## Installation

Add the following dependency to your `Cargo.toml`:

```toml
[dependencies]
bevy_radix_sort = "0.1.0"
```

## Benchmark

todo!

## Usage

```rust
todo!()
```

## Contributing

Contributions are welcome! Feel free to submit issues or PRs.