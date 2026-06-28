/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-instruction warp floating-point reductions (`redux.sync.fmin/fmax`,
//! datacenter Blackwell sm_100a).
//!
//! Companion to `redux_sum` and `redux_minmax`. This example exercises the
//! Blackwell f32 min/max warp reductions, lowered to one hardware `redux.sync.*`
//! instruction instead of a `shfl`-based log-tree.
//!
//! NOTE: `redux.sync.f32` is only available on datacenter Blackwell (B100/B200,
//! sm_100a). Consumer Blackwell (RTX 50 series, sm_120) does not support it, so
//! the host code skips the kernel launch on sm_120.
//!
//! Build and run with:
//!   cargo oxide run redux_fminmax --arch sm_100a

use cuda_device::{DisjointSlice, kernel, warp};
use cuda_host::cuda_module;

const FULL_MASK: u32 = 0xffff_ffff;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Lane `l` contributes `l as f32 - 15.5`, i.e. the values `-15.5..=15.5`.
    ///
    /// The warp-wide min and max are broadcast back to every lane; lane 0 writes
    /// `[fmin, fmax]`.
    #[kernel]
    pub fn redux_fminmax(mut out: DisjointSlice<f32>) {
        let lane = warp::lane_id();
        let v = lane as f32 - 15.5;

        let fmin = warp::redux_sync_fmin(FULL_MASK, v);
        let fmax = warp::redux_sync_fmax(FULL_MASK, v);

        if lane == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = fmin;
                *out.get_unchecked_mut(1) = fmax;
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

    println!("=== redux.sync.fmin/fmax (datacenter Blackwell sm_100a) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // `redux.sync.f32` requires datacenter Blackwell (sm_100a). It is NOT
    // available on consumer Blackwell (RTX 50 series, sm_120).
    if major < 10 {
        println!("\nskipping: redux.sync.fmin/fmax requires Blackwell (sm_100a+)");
        println!("  this GPU is sm_{}{}", major, minor);
        return;
    }
    if major == 12 {
        println!("\nskipping: redux.sync.f32 is not available on consumer Blackwell (sm_120)");
        println!("  it requires datacenter Blackwell (B100/B200, sm_100a)");
        println!("  this GPU is sm_{}{}", major, minor);
        return;
    }

    let module = ctx
        .load_module_from_file("redux_fminmax.ptx")
        .expect("Failed to load PTX module");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    // A single warp is all we need to demonstrate the reduction semantics.
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("\n--- Test: redux.sync.fmin/fmax ---");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, 2).unwrap();

    module
        .redux_fminmax((stream).as_ref(), cfg, &mut out_dev)
        .expect("Kernel launch failed");

    let out = out_dev.to_host_vec(&stream).unwrap();
    println!("[fmin, fmax] = {:?}        (expected [-15.5, 15.5])", out);

    if out[0] == -15.5 && out[1] == 15.5 {
        println!("✓ fmin/fmax both correct");
    } else {
        println!("✗ fmin/fmax mismatch!");
        std::process::exit(1);
    }

    println!("\nSUCCESS: redux.sync.fmin/fmax produced correct results");
}
