/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

//! Flash Attention v2 Forward Pass (single-head, FP32)
//!
//! Implements the online-softmax tiled attention algorithm from
//! "FlashAttention-2: Faster Attention with Better Parallelism and Work
//! Partitioning" (Tri Dao, 2023).
//!
//! Given Q, K, V of shape `[N, D]`, computes
//!     O = softmax(Q @ K^T / sqrt(D)) @ V
//! without ever materializing the `N x N` attention matrix in global memory.
//!
//! Layout:
//! - One CTA per `BR`-row tile of Q (outer loop over Q rows)
//! - Inner loop streams `BC`-row tiles of K, V through shared memory
//! - Online softmax: each iteration updates the running row max `m_i`,
//!   normalizer `l_i`, and output accumulator `O_i`
//!
//! Build and run with:
//!   cargo oxide run flash_attention

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use std::time::Instant;

// =============================================================================
// Tile constants (compile-time)
// =============================================================================
//
// Smem footprint per CTA:
//   Q_TILE  BR*D = 32*64 = 2048 f32  =  8 KB
//   K_TILE  BC*D = 32*64 = 2048 f32  =  8 KB
//   V_TILE  BC*D = 32*64 = 2048 f32  =  8 KB
//   S_TILE  BR*BC = 32*32 = 1024 f32  =  4 KB
//   O_TILE  BR*D = 32*64 = 2048 f32  =  8 KB
//   total = 36 KB  (fits in the default 48 KB smem on Ampere+)
const D: usize = 64;
const BR: usize = 32;
const BC: usize = 32;

// SharedArray needs literal const sizes today; assert the arithmetic.
const _: () = assert!(BR * D == 2048);
const _: () = assert!(BC * D == 2048);
const _: () = assert!(BR * BC == 1024);

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Naive single-pass attention. One thread per output element.
    ///
    /// Reference implementation for correctness checking. Materializes one row
    /// of `S = Q @ K^T / sqrt(D)` in registers, softmaxes it, then dots against
    /// V. No tiling, no online softmax — just the textbook formula. Slow for
    /// large N, but unambiguous.
    ///
    /// Launch:  grid=(N, 1, 1), block=(D, 1, 1)
    /// Each block handles one query row; each thread handles one output column.
    #[kernel]
    pub fn attention_naive(
        seq_len: u32,
        scale: f32,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut o: DisjointSlice<f32>,
    ) {
        let n = seq_len as usize;
        let row = thread::blockIdx_x() as usize;
        let col = thread::threadIdx_x() as usize;

        if row >= n || col >= D {
            return;
        }

        // First pass: compute S_row = (Q[row] . K[j]) * scale for j in 0..n.
        // Keep the max for the stable softmax. Each thread does this redundantly
        // because we need the full S_row to softmax against, but that keeps the
        // example simple — this is the *naive* baseline.
        let mut row_max = -1.0e30f32;
        let mut j = 0usize;
        while j < n {
            let mut s = 0.0f32;
            let mut kk = 0usize;
            while kk < D {
                s += q[row * D + kk] * k[j * D + kk];
                kk += 1;
            }
            s *= scale;
            if s > row_max {
                row_max = s;
            }
            j += 1;
        }

        // Second pass: compute sum of exp(S_row - max).
        let mut sum_exp = 0.0f32;
        j = 0;
        while j < n {
            let mut s = 0.0f32;
            let mut kk = 0usize;
            while kk < D {
                s += q[row * D + kk] * k[j * D + kk];
                kk += 1;
            }
            sum_exp += (s * scale - row_max).exp();
            j += 1;
        }

        // Third pass: this thread accumulates O[row, col] = sum_j P[row,j] * V[j, col].
        let mut acc = 0.0f32;
        j = 0;
        while j < n {
            let mut s = 0.0f32;
            let mut kk = 0usize;
            while kk < D {
                s += q[row * D + kk] * k[j * D + kk];
                kk += 1;
            }
            let p = (s * scale - row_max).exp() / sum_exp;
            acc += p * v[j * D + col];
            j += 1;
        }

        unsafe {
            *o.get_unchecked_mut(row * D + col) = acc;
        }
    }

    /// Flash Attention v2 forward pass. Tiled, online-softmax.
    ///
    /// Launch:  grid=(N/BR, 1, 1), block=(BR, 1, 1)
    /// Each CTA owns `BR` rows of Q and produces `BR` rows of O. Inner loop
    /// streams `BC`-row tiles of K, V through shared memory.
    ///
    /// Per-thread loop invariant after iteration j:
    ///   m_i = max over all keys seen so far of S[row, k]
    ///   l_i = sum over keys seen of exp(S[row, k] - m_i)
    ///   O_TILE[row] = (unnormalized) sum over keys seen of exp(S - m_i) * V[k]
    /// After the final iteration we divide by l_i.
    #[kernel]
    pub fn flash_attention_v2(
        seq_len: u32,
        scale: f32,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut o: DisjointSlice<f32>,
    ) {
        static mut Q_TILE: SharedArray<f32, 2048> = SharedArray::UNINIT; // BR * D
        static mut K_TILE: SharedArray<f32, 2048> = SharedArray::UNINIT; // BC * D
        static mut V_TILE: SharedArray<f32, 2048> = SharedArray::UNINIT; // BC * D
        static mut S_TILE: SharedArray<f32, 1024> = SharedArray::UNINIT; // BR * BC
        static mut O_TILE: SharedArray<f32, 2048> = SharedArray::UNINIT; // BR * D

        let n = seq_len as usize;
        let tid = thread::threadIdx_x() as usize; // 0..BR
        let tile_i = thread::blockIdx_x() as usize; // which Q tile
        let q_row = tile_i * BR + tid; // global Q row owned by this thread

        // Cooperatively load this CTA's Q tile and zero-init its O tile.
        // The CTA has BR threads; the tile has BR*D entries. Each thread loads
        // exactly D entries (its own row) — coalesced because consecutive
        // threads load consecutive Q rows but each thread's D loads are
        // contiguous within the row.
        if q_row < n {
            unsafe {
                let mut kk = 0usize;
                while kk < D {
                    Q_TILE[tid * D + kk] = q[q_row * D + kk];
                    O_TILE[tid * D + kk] = 0.0;
                    kk += 1;
                }
            }
        }

        // Per-thread running softmax statistics for this Q row.
        let mut m_i: f32 = -1.0e30;
        let mut l_i: f32 = 0.0;

        thread::sync_threads();

        // Outer loop over K, V tiles.
        let num_kv_tiles = n / BC;
        let mut j_tile = 0usize;
        while j_tile < num_kv_tiles {
            // Cooperative load: each thread strides through BC*D entries.
            // With BR=BC=32 and D=64, each thread loads 64 floats per matrix.
            let base = j_tile * BC;
            unsafe {
                let mut idx = tid;
                while idx < BC * D {
                    let r = idx / D;
                    let c = idx % D;
                    let global_row = base + r;
                    K_TILE[idx] = k[global_row * D + c];
                    V_TILE[idx] = v[global_row * D + c];
                    idx += BR;
                }
            }
            thread::sync_threads();

            if q_row < n {
                // Compute this thread's row of S_ij = Q_i @ K_j^T * scale
                // and track its row max.
                let mut s_row_max = -1.0e30f32;
                unsafe {
                    let mut c = 0usize;
                    while c < BC {
                        let mut acc = 0.0f32;
                        let mut kk = 0usize;
                        while kk < D {
                            acc += Q_TILE[tid * D + kk] * K_TILE[c * D + kk];
                            kk += 1;
                        }
                        acc *= scale;
                        S_TILE[tid * BC + c] = acc;
                        if acc > s_row_max {
                            s_row_max = acc;
                        }
                        c += 1;
                    }
                }

                // Online softmax update.
                let m_new = if m_i > s_row_max { m_i } else { s_row_max };
                let correction = (m_i - m_new).exp();

                // P_ij = exp(S - m_new); l_tilde = sum P_ij.
                let mut l_tilde = 0.0f32;
                unsafe {
                    let mut c = 0usize;
                    while c < BC {
                        let p = (S_TILE[tid * BC + c] - m_new).exp();
                        S_TILE[tid * BC + c] = p;
                        l_tilde += p;
                        c += 1;
                    }
                }

                // O_new = correction * O_old + P_ij @ V_j
                unsafe {
                    let mut kk = 0usize;
                    while kk < D {
                        let mut acc = correction * O_TILE[tid * D + kk];
                        let mut c = 0usize;
                        while c < BC {
                            acc += S_TILE[tid * BC + c] * V_TILE[c * D + kk];
                            c += 1;
                        }
                        O_TILE[tid * D + kk] = acc;
                        kk += 1;
                    }
                }

                l_i = correction * l_i + l_tilde;
                m_i = m_new;
            }

            // Wait before next iteration overwrites K_TILE / V_TILE.
            thread::sync_threads();
            j_tile += 1;
        }

        // Normalize and write out.
        if q_row < n {
            let inv_l = 1.0f32 / l_i;
            unsafe {
                let mut kk = 0usize;
                while kk < D {
                    let val = O_TILE[tid * D + kk] * inv_l;
                    *o.get_unchecked_mut(q_row * D + kk) = val;
                    kk += 1;
                }
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

const N: usize = 1024; // sequence length
const NUM_RUNS: u32 = 20;

/// CPU reference attention: O = softmax(Q @ K^T / sqrt(D)) @ V.
/// Computes one row at a time to keep memory usage bounded.
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], n: usize, d: usize) -> Vec<f32> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let mut o = vec![0.0f32; n * d];
    let mut s = vec![0.0f32; n];

    for i in 0..n {
        // s[j] = (Q[i] . K[j]) * scale
        let mut row_max = f32::NEG_INFINITY;
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..d {
                acc += q[i * d + kk] * k[j * d + kk];
            }
            acc *= scale;
            s[j] = acc;
            if acc > row_max {
                row_max = acc;
            }
        }
        // softmax in place
        let mut sum_exp = 0.0f32;
        for j in 0..n {
            s[j] = (s[j] - row_max).exp();
            sum_exp += s[j];
        }
        let inv_sum = 1.0f32 / sum_exp;
        // O[i] = s @ V
        for kk in 0..d {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += s[j] * v[j * d + kk];
            }
            o[i * d + kk] = acc * inv_sum;
        }
    }
    o
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    m
}

fn main() {
    println!("=== Flash Attention v2 Forward Pass ===");
    println!("N = {}, D = {}, BR = {}, BC = {}", N, D, BR, BC);
    println!("Smem per CTA: 36 KB (Q+K+V+S+O tiles)\n");

    assert_eq!(N % BR, 0, "N must be divisible by BR");
    assert_eq!(N % BC, 0, "N must be divisible by BC");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // Reproducible pseudo-random inputs. Stay small to keep S in a friendly
    // range for FP32 (no overflow in exp).
    println!("Generating inputs...");
    let mut q = vec![0.0f32; N * D];
    let mut k = vec![0.0f32; N * D];
    let mut v = vec![0.0f32; N * D];
    let mut seed: u64 = 0xdeadbeef;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 as f32) * (1.0 / (1u64 << 31) as f32) * 0.5
    };
    for v_buf in [&mut q, &mut k, &mut v] {
        for x in v_buf.iter_mut() {
            *x = rand();
        }
    }

    let scale = 1.0f32 / (D as f32).sqrt();
    println!("Scale = 1/sqrt(D) = {:.6}\n", scale);

    let q_dev = DeviceBuffer::from_host(&stream, &q).unwrap();
    let k_dev = DeviceBuffer::from_host(&stream, &k).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, &v).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    // -------------------------------------------------------------------------
    // CPU reference
    // -------------------------------------------------------------------------
    println!("Computing CPU reference (this takes a few seconds)...");
    let t = Instant::now();
    let cpu_out = cpu_attention(&q, &k, &v, N, D);
    let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("CPU: {:.1} ms\n", cpu_ms);

    // -------------------------------------------------------------------------
    // Naive GPU kernel
    // -------------------------------------------------------------------------
    println!("--- Naive GPU attention ---");
    let mut naive_out = DeviceBuffer::<f32>::zeroed(&stream, N * D).unwrap();
    let naive_cfg = LaunchConfig {
        grid_dim: (N as u32, 1, 1),
        block_dim: (D as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warmup
    module
        .attention_naive(
            &stream,
            naive_cfg,
            N as u32,
            scale,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut naive_out,
        )
        .expect("naive launch failed");
    stream.synchronize().unwrap();

    let t = Instant::now();
    for _ in 0..NUM_RUNS {
        module
            .attention_naive(
                &stream,
                naive_cfg,
                N as u32,
                scale,
                &q_dev,
                &k_dev,
                &v_dev,
                &mut naive_out,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    let naive_ms = t.elapsed().as_secs_f64() * 1000.0 / NUM_RUNS as f64;
    let naive_host = naive_out.to_host_vec(&stream).unwrap();
    let naive_err = max_abs_diff(&naive_host, &cpu_out);
    println!("Naive GPU: {:.3} ms/iter, max |err| vs CPU = {:.3e}", naive_ms, naive_err);

    // -------------------------------------------------------------------------
    // Flash Attention v2
    // -------------------------------------------------------------------------
    println!("\n--- Flash Attention v2 ---");
    let mut flash_out = DeviceBuffer::<f32>::zeroed(&stream, N * D).unwrap();
    let flash_cfg = LaunchConfig {
        grid_dim: ((N / BR) as u32, 1, 1),
        block_dim: (BR as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warmup
    module
        .flash_attention_v2(
            &stream,
            flash_cfg,
            N as u32,
            scale,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut flash_out,
        )
        .expect("flash launch failed");
    stream.synchronize().unwrap();

    let t = Instant::now();
    for _ in 0..NUM_RUNS {
        module
            .flash_attention_v2(
                &stream,
                flash_cfg,
                N as u32,
                scale,
                &q_dev,
                &k_dev,
                &v_dev,
                &mut flash_out,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    let flash_ms = t.elapsed().as_secs_f64() * 1000.0 / NUM_RUNS as f64;
    let flash_host = flash_out.to_host_vec(&stream).unwrap();
    let flash_err = max_abs_diff(&flash_host, &cpu_out);
    println!("Flash v2:  {:.3} ms/iter, max |err| vs CPU = {:.3e}", flash_ms, flash_err);

    // -------------------------------------------------------------------------
    // Throughput
    // -------------------------------------------------------------------------
    // Two matmuls of dimension (N, D) x (D, N) and (N, N) x (N, D):
    //   FLOPs = 2 * N * N * D + 2 * N * N * D = 4 * N^2 * D
    let flops = 4.0 * (N as f64) * (N as f64) * (D as f64);
    let naive_tflops = flops / (naive_ms / 1000.0) / 1e12;
    let flash_tflops = flops / (flash_ms / 1000.0) / 1e12;

    println!("\n=== Summary ===");
    println!("Naive GPU: {:>8.3} ms  ({:>5.2} TFLOPS)", naive_ms, naive_tflops);
    println!("Flash v2:  {:>8.3} ms  ({:>5.2} TFLOPS)   {:.2}x vs naive",
        flash_ms, flash_tflops, naive_ms / flash_ms);

    // Tolerances are generous because attention involves exp, division, and
    // accumulation over N=1024 terms; FP32 round-off easily reaches 1e-4.
    let tol = 5.0e-4;
    if naive_err < tol && flash_err < tol {
        println!("\n✓ SUCCESS: both kernels match CPU within {:.0e}", tol);
    } else {
        println!("\n✗ FAILED: tolerance {:.0e} exceeded", tol);
        std::process::exit(1);
    }
}
