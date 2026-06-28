/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Positive test: enum discriminant writes via `SetDiscriminant` are lowered
//! correctly on the device.
//!
//! The custom MIR helper below emits `StatementKind::SetDiscriminant`.
//! After lowering, the enum's tag is updated in memory and the kernel can
//! observe the new variant.
//!
//! Usage:
//!   cargo oxide run set_discriminant

#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]

use core::intrinsics::mir::*;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[allow(dead_code)]
enum DeviceState {
    Empty,
    Full(u32),
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn force_set_discriminant(state: &mut DeviceState) {
    mir!({
        SetDiscriminant(*state, 0);
        Return()
    })
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Each thread starts with `DeviceState::Full(idx)`, then uses custom MIR
    /// to emit `SetDiscriminant` to `Empty`. The output is `1` if the
    /// discriminant write was observed, `0` otherwise.
    #[kernel]
    pub fn set_discriminant_kernel(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let raw_idx = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut state = DeviceState::Full(raw_idx as u32);

            // This helper emits `StatementKind::SetDiscriminant` directly.
            force_set_discriminant(&mut state);

            *out_elem = match state {
                DeviceState::Empty => 1,
                DeviceState::Full(_) => 0,
            };
        }
    }
}

fn main() {
    println!("=== set_discriminant ===");
    println!("Verifying that MIR SetDiscriminant is lowered to a device-side tag write.");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 64;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .set_discriminant_kernel(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev)
        .expect("Kernel launch failed");

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    let mut errors = 0;
    for (i, &v) in out_host.iter().enumerate() {
        if v != 1 {
            errors += 1;
            if errors <= 5 {
                eprintln!("  Error at [{}]: expected 1 (Empty), got {}", i, v);
            }
        }
    }

    if errors == 0 {
        println!(
            "PASS: all {} threads observed the SetDiscriminant write.",
            N
        );
    } else {
        eprintln!(
            "FAIL: {} threads did not observe the SetDiscriminant write.",
            errors
        );
        std::process::exit(1);
    }
}
