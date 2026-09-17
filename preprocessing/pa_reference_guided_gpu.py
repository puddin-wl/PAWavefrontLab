"""High-throughput CUDA pipeline for reference-guided PA denoising.

This version is tuned for large RTX GPUs and repeated 600x600x512 volumes.
It keeps the validated signal-processing math unchanged while reducing the
host/device and memory-allocation gaps between chunks:

* packed-12 input is read into page-locked (pinned) host buffers;
* two packed-input buffers are ping-ponged so H2D of chunk N+1 overlaps CUDA
  processing of chunk N;
* packed-12 decoding writes directly to a reusable float32 GPU A-line buffer;
* the historical NCC + least-squares subtraction stays entirely on CUDA;
* QC robust detrending computes only the final signal-window residual;
* excess-RMS computes only the two noise windows plus the signal window rather
  than materialising a full-depth detrended copy;
* only the final 1-D projection and three scalar QC values return to the CPU;
* post-FFT match selection and shifted-template subtraction use fused RawKernels
  with persistent result buffers, avoiding dynamic flatnonzero/shifted arrays;
* row-wise medians use one shared-memory RawKernel instead of CuPy's generic
  partition path, removing its many internal launches/synchronizations.

CuPy is imported lazily through :mod:`preprocessing.pa_denoising_gpu` so CPU-
only environments can still import the project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preprocessing.pa_denoising_gpu import (
    _correlation_terms_gpu,
    _require_cupy,
    _template_energy_cpu,
)


@dataclass(frozen=True)
class GpuChunkResult:
    projection: np.ndarray
    matched_count: int
    signal_power_before: float
    signal_power_after: float


@dataclass
class PackedTransferSlot:
    """One pinned-host/device packed-byte ping-pong slot."""

    host_memory: object
    host: np.ndarray
    device: object
    copy_stream: object
    ready_event: object
    capacity_bytes: int
    valid_bytes: int = 0


class ReferenceGuidedGpuPipeline:
    """Persistent CUDA workspace shared by every volume in one experiment.

    The expensive CUDA context, template constants, pinned transfer slots, decoded
    float32 buffer and CuPy/FFT memory-pool high-water mark stay alive across
    origin + all measurement volumes.  Call :meth:`shutdown` once after the full
    experiment rather than tearing the workspace down after every BIN file.
    """

    def __init__(self, calibration) -> None:
        cp, fftconvolve = _require_cupy()
        self.cp = cp
        self.fftconvolve = fftconvolve
        self.device_id = int(cp.cuda.Device().id)
        self.depth = int(calibration.depth)
        self.signal_window = tuple(int(v) for v in calibration.signal_window)
        self.noise_windows = tuple(
            tuple(int(v) for v in window) for window in calibration.noise_windows
        )
        self.correlation_threshold = float(calibration.template.correlation_threshold)
        self.minimum_fitted_peak_adc = float(
            calibration.template.minimum_fitted_peak_adc
        )

        kernel_np = np.asarray(calibration.template.values, dtype=np.float32)
        if kernel_np.ndim != 1 or kernel_np.size < 3 or kernel_np.size % 2 != 1:
            raise ValueError("template 必须是一维奇数长度数组。")
        if not np.isfinite(kernel_np).all() or float(np.max(np.abs(kernel_np))) <= 0:
            raise ValueError("template 必须为有限非零数组。")

        self.kernel = cp.asarray(kernel_np, dtype=cp.float32)
        self.kernel_peak_host = np.float32(np.max(np.abs(kernel_np)))
        self.kernel_peak = cp.float32(self.kernel_peak_host)
        self.template_energy = cp.asarray(
            _template_energy_cpu(kernel_np, self.depth), dtype=cp.float32
        )
        self.half_width = int(kernel_np.size // 2)
        self.depth_indices = cp.arange(self.depth, dtype=cp.int32)
        self.full_positions = cp.arange(self.depth, dtype=cp.float32)

        left, right = self.noise_windows
        noise_indices_np = np.r_[left[0] : left[1], right[0] : right[1]].astype(
            np.int32
        )
        self.noise_indices = cp.asarray(noise_indices_np, dtype=cp.int32)
        self.noise_positions = self.noise_indices.astype(cp.float32)
        self.left_positions = self.full_positions[left[0] : left[1]]
        self.right_positions = self.full_positions[right[0] : right[1]]
        self.signal_positions = self.full_positions[
            self.signal_window[0] : self.signal_window[1]
        ]

        left_center = np.float32((left[0] + left[1] - 1) / 2.0)
        right_center = np.float32((right[0] + right[1] - 1) / 2.0)
        signal_positions_np = np.arange(
            self.signal_window[0], self.signal_window[1], dtype=np.float32
        )
        interpolation_np = (signal_positions_np - left_center) / max(
            float(right_center - left_center), 1.0
        )
        self.interpolation = cp.asarray(interpolation_np, dtype=cp.float32)
        self.out_of_memory_error = cp.cuda.memory.OutOfMemoryError

        # One compute stream plus two independent copy streams.  Compute remains
        # serial (important for 16 GB cards); only packed H2D for the next chunk
        # overlaps current CUDA work.
        self.compute_stream = cp.cuda.Stream(non_blocking=True)
        self.transfer_slots: tuple[PackedTransferSlot, ...] = ()
        self._values_buffer = None
        self._centers_buffer = None
        self._coefficients_buffer = None
        self._matched_buffer = None
        self._median_buffer0 = None
        self._median_buffer1 = None
        self._host_output_memory = None
        self._host_projection = None
        self._host_stats = None
        self._max_alines = 0
        self._max_packed_bytes = 0
        self._closed = False

        # Direct packed-u12 -> float32 decode avoids a temporary uint16 volume
        # and a second conversion kernel/copy.
        self._decode_u12_f32 = cp.RawKernel(
            r'''
            extern "C" __global__
            void decode_u12_f32(
                const unsigned char* packed,
                float* output,
                const long long pair_count)
            {
                const long long i =
                    (long long)blockDim.x * blockIdx.x + threadIdx.x;
                if (i >= pair_count) return;
                const unsigned char b0 = packed[3 * i + 0];
                const unsigned char b1 = packed[3 * i + 1];
                const unsigned char b2 = packed[3 * i + 2];
                const unsigned short first =
                    (unsigned short)b0 | ((unsigned short)(b1 & 0x0F) << 8);
                const unsigned short second =
                    ((unsigned short)b1 >> 4) | ((unsigned short)b2 << 4);
                output[2 * i + 0] = (float)first;
                output[2 * i + 1] = (float)second;
            }
            ''',
            "decode_u12_f32",
        )

        # Row-wise median specialized for <=512 samples.  CuPy's generic
        # median/partition implementation can internally launch many kernels and
        # synchronize repeatedly.  One CUDA block now owns one A-line/window and
        # sorts its values in shared memory, so each row-median call is a single
        # asynchronous kernel launch on the experiment compute stream.
        self._row_median_kernel = cp.RawKernel(
            r'''
            extern "C" __global__
            void row_median_512(
                const float* input,
                float* output,
                const int rows,
                const int cols,
                const long long row_stride,
                const long long col_stride,
                const int sort_size)
            {
                const int row = blockIdx.x;
                const int tid = threadIdx.x;
                if (row >= rows) return;

                __shared__ float data[512];
                const long long base = (long long)row * row_stride;

                // 256 threads load up to 512 elements.  Pad to the next power
                // of two with +inf so valid samples remain in data[0:cols]
                // after the ascending bitonic sort.
                int i = tid;
                if (i < sort_size) {
                    data[i] = (i < cols)
                        ? input[base + (long long)i * col_stride]
                        : 3.402823466e+38F;
                }
                i = tid + 256;
                if (i < sort_size) {
                    data[i] = (i < cols)
                        ? input[base + (long long)i * col_stride]
                        : 3.402823466e+38F;
                }
                __syncthreads();

                for (int k = 2; k <= sort_size; k <<= 1) {
                    for (int j = k >> 1; j > 0; j >>= 1) {
                        int idx = tid;
                        if (idx < sort_size) {
                            const int partner = idx ^ j;
                            if (partner > idx && partner < sort_size) {
                                const bool ascending = ((idx & k) == 0);
                                const float a = data[idx];
                                const float b = data[partner];
                                if ((a > b) == ascending) {
                                    data[idx] = b;
                                    data[partner] = a;
                                }
                            }
                        }
                        idx = tid + 256;
                        if (idx < sort_size) {
                            const int partner = idx ^ j;
                            if (partner > idx && partner < sort_size) {
                                const bool ascending = ((idx & k) == 0);
                                const float a = data[idx];
                                const float b = data[partner];
                                if ((a > b) == ascending) {
                                    data[idx] = b;
                                    data[partner] = a;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }

                if (tid == 0) {
                    if (cols & 1) {
                        output[row] = data[cols >> 1];
                    } else {
                        const int hi = cols >> 1;
                        output[row] = 0.5f * (data[hi - 1] + data[hi]);
                    }
                }
            }
            ''',
            "row_median_512",
        )

        # Fuse amplitude gating + two argmax reductions + LS coefficient lookup
        # into one kernel.  The previous CuPy expression chain materialised
        # coefficient/fitted-peak/valid/score arrays and used several reductions.
        # Keeping only correlation+dot from the FFT stage substantially reduces
        # launches, temporary memory traffic and dynamic-shape synchronization.
        self._select_match_kernel = cp.RawKernel(
            r'''
            extern "C" __global__
            void select_match(
                const float* correlation,
                const float* dot,
                const float* template_energy,
                int* centers,
                float* coefficients,
                unsigned char* matched,
                const int rows,
                const int depth,
                const long long correlation_row_stride,
                const long long dot_row_stride,
                const float kernel_peak,
                const float amplitude_threshold,
                const float correlation_threshold)
            {
                const int row = blockIdx.x;
                const int tid = threadIdx.x;
                if (row >= rows) return;

                __shared__ float valid_score[256];
                __shared__ float global_score[256];
                __shared__ int valid_index[256];
                __shared__ int global_index[256];

                float local_valid = -1.0f;
                float local_global = -1.0f;
                int local_valid_idx = 0x7fffffff;
                int local_global_idx = 0x7fffffff;

                // correlation/dot may be strided views (notably dot is sliced
                // from a full FFT-convolution result).  Never assume row stride
                // equals depth inside a RawKernel.
                const long long corr_base = (long long)row * correlation_row_stride;
                const long long dot_base = (long long)row * dot_row_stride;
                for (int c = tid; c < depth; c += blockDim.x) {
                    const float corr = correlation[corr_base + c];
                    const float abs_corr = fabsf(corr);
                    const float energy = fmaxf(template_energy[c], 1.0e-12f);
                    const float coeff = dot[dot_base + c] / energy;
                    const bool amp_ok = fabsf(coeff) * kernel_peak >= amplitude_threshold;

                    if (abs_corr > local_global ||
                        (abs_corr == local_global && c < local_global_idx)) {
                        local_global = abs_corr;
                        local_global_idx = c;
                    }
                    if (amp_ok &&
                        (abs_corr > local_valid ||
                         (abs_corr == local_valid && c < local_valid_idx))) {
                        local_valid = abs_corr;
                        local_valid_idx = c;
                    }
                }

                valid_score[tid] = local_valid;
                global_score[tid] = local_global;
                valid_index[tid] = local_valid_idx;
                global_index[tid] = local_global_idx;
                __syncthreads();

                for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        const float ov = valid_score[tid + stride];
                        const int ovi = valid_index[tid + stride];
                        if (ov > valid_score[tid] ||
                            (ov == valid_score[tid] && ovi < valid_index[tid])) {
                            valid_score[tid] = ov;
                            valid_index[tid] = ovi;
                        }

                        const float og = global_score[tid + stride];
                        const int ogi = global_index[tid + stride];
                        if (og > global_score[tid] ||
                            (og == global_score[tid] && ogi < global_index[tid])) {
                            global_score[tid] = og;
                            global_index[tid] = ogi;
                        }
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    const bool has_valid = valid_score[0] >= 0.0f;
                    const int center = has_valid ? valid_index[0] : global_index[0];
                    const float energy = fmaxf(template_energy[center], 1.0e-12f);
                    const float coeff = dot[dot_base + center] / energy;
                    centers[row] = center;
                    coefficients[row] = coeff;
                    matched[row] = (unsigned char)(
                        has_valid && fabsf(correlation[corr_base + center]) >= correlation_threshold
                    );
                }
            }
            ''',
            "select_match",
        )

        # Subtract the selected shifted template directly in-place.  This replaces
        # flatnonzero + advanced indexing + full shifted-template construction.
        self._subtract_selected_kernel = cp.RawKernel(
            r'''
            extern "C" __global__
            void subtract_selected(
                float* values,
                const float* kernel,
                const int kernel_size,
                const int half_width,
                const int* centers,
                const float* coefficients,
                const unsigned char* matched,
                const long long total_values,
                const int depth)
            {
                const long long i =
                    (long long)blockDim.x * blockIdx.x + threadIdx.x;
                if (i >= total_values) return;
                const int row = (int)(i / depth);
                if (!matched[row]) return;
                const int col = (int)(i - (long long)row * depth);
                const int ti = col - centers[row] + half_width;
                if (ti >= 0 && ti < kernel_size) {
                    values[i] -= coefficients[row] * kernel[ti];
                }
            }
            ''',
            "subtract_selected",
        )

    def release_cached_memory(self) -> None:
        """Release cached CuPy/FFT blocks; persistent pipeline buffers remain."""
        cp = self.cp
        try:
            self.compute_stream.synchronize()
        except Exception:
            pass
        for slot in self.transfer_slots:
            try:
                slot.copy_stream.synchronize()
            except Exception:
                pass
        cp.get_default_memory_pool().free_all_blocks()
        try:
            cp.fft.config.get_plan_cache().clear()
        except Exception:
            pass

    def synchronize(self) -> None:
        """Wait for all pipeline streams without releasing persistent buffers."""
        if self._closed:
            return
        try:
            self.compute_stream.synchronize()
        except Exception:
            pass
        for slot in self.transfer_slots:
            try:
                slot.copy_stream.synchronize()
            except Exception:
                pass

    def shutdown(self) -> None:
        """Release the experiment-level CUDA workspace exactly once."""
        if self._closed:
            return
        self.synchronize()
        self.transfer_slots = ()
        self._values_buffer = None
        self._centers_buffer = None
        self._coefficients_buffer = None
        self._matched_buffer = None
        self._median_buffer0 = None
        self._median_buffer1 = None
        self._host_output_memory = None
        self._host_projection = None
        self._host_stats = None
        self._max_alines = 0
        self._max_packed_bytes = 0
        try:
            self.cp.fft.config.get_plan_cache().clear()
        except Exception:
            pass
        self.cp.get_default_memory_pool().free_all_blocks()
        self.cp.get_default_pinned_memory_pool().free_all_blocks()
        self._closed = True

    # Backwards-compatible alias.  New experiment runners should call shutdown()
    # only after every origin/measurement volume has finished.
    def close_streaming(self) -> None:
        self.shutdown()

    def device_info(self) -> dict:
        cp = self.cp
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return {
            "gpu_name": str(name),
            "free_memory_gib": float(free_bytes / (1024**3)),
            "total_memory_gib": float(total_bytes / (1024**3)),
            "streaming_buffers": len(self.transfer_slots),
            "max_alines": int(self._max_alines),
        }

    def prepare_streaming(
        self,
        *,
        max_packed_bytes: int,
        max_alines: int,
        slot_count: int = 2,
    ) -> tuple[PackedTransferSlot, ...]:
        """Allocate reusable pinned/device input buffers for repeated chunks."""
        if max_packed_bytes <= 0 or max_alines <= 0:
            raise ValueError("max_packed_bytes 和 max_alines 必须为正数。")
        if self._closed:
            raise RuntimeError("GPU pipeline 已 shutdown；请为新实验创建新实例。")
        if slot_count != 2:
            raise ValueError("当前流水线固定使用 2 个 ping-pong slot。")
        if (
            self.transfer_slots
            and self._max_packed_bytes >= max_packed_bytes
            and self._max_alines >= max_alines
        ):
            return self.transfer_slots

        # Resize only when the requested experiment chunk is larger than the
        # existing workspace.  Do not clear the CuPy memory pool here: retaining
        # its high-water mark is what removes per-volume allocation sawteeth.
        self.synchronize()
        self.transfer_slots = ()
        self._values_buffer = None
        self._centers_buffer = None
        self._coefficients_buffer = None
        self._matched_buffer = None
        self._median_buffer0 = None
        self._median_buffer1 = None
        self._host_output_memory = None
        self._host_projection = None
        self._host_stats = None
        self._max_alines = 0
        self._max_packed_bytes = 0
        self._closed = False
        cp = self.cp
        slots = []
        for _ in range(slot_count):
            host_memory = cp.cuda.alloc_pinned_memory(max_packed_bytes)
            host = np.frombuffer(
                host_memory, dtype=np.uint8, count=max_packed_bytes
            )
            device = cp.empty(max_packed_bytes, dtype=cp.uint8)
            copy_stream = cp.cuda.Stream(non_blocking=True)
            ready_event = cp.cuda.Event()
            slots.append(
                PackedTransferSlot(
                    host_memory=host_memory,
                    host=host,
                    device=device,
                    copy_stream=copy_stream,
                    ready_event=ready_event,
                    capacity_bytes=max_packed_bytes,
                )
            )

        # Only one decoded float32 volume is required because compute itself is
        # serial; the next chunk overlaps only at packed-byte H2D stage.
        self._values_buffer = cp.empty(
            max_alines * self.depth, dtype=cp.float32
        )
        self._centers_buffer = cp.empty(max_alines, dtype=cp.int32)
        self._coefficients_buffer = cp.empty(max_alines, dtype=cp.float32)
        self._matched_buffer = cp.empty(max_alines, dtype=cp.uint8)
        self._median_buffer0 = cp.empty(max_alines, dtype=cp.float32)
        self._median_buffer1 = cp.empty(max_alines, dtype=cp.float32)

        # One pinned return slab holds the 1-D projection plus three float64 QC
        # values.  Both D2H copies are enqueued, followed by one stream sync.
        output_bytes = max_alines * np.dtype(np.float32).itemsize + 3 * np.dtype(np.float64).itemsize
        self._host_output_memory = cp.cuda.alloc_pinned_memory(output_bytes)
        raw_output = np.frombuffer(self._host_output_memory, dtype=np.uint8, count=output_bytes)
        projection_bytes = max_alines * np.dtype(np.float32).itemsize
        self._host_projection = raw_output[:projection_bytes].view(np.float32)
        self._host_stats = raw_output[projection_bytes:].view(np.float64)

        self._max_alines = int(max_alines)
        self._max_packed_bytes = int(max_packed_bytes)
        self.transfer_slots = tuple(slots)
        return self.transfer_slots

    def enqueue_h2d(self, slot: PackedTransferSlot, *, nbytes: int) -> None:
        """Asynchronously copy one pinned packed-byte buffer to its GPU slot."""
        if nbytes <= 0 or nbytes > slot.capacity_bytes:
            raise ValueError(
                f"nbytes 必须位于 1..{slot.capacity_bytes}，实际为 {nbytes}。"
            )
        slot.valid_bytes = int(nbytes)
        with self.cp.cuda.Device(self.device_id):
            slot.device[:nbytes].set(
                slot.host[:nbytes], stream=slot.copy_stream
            )
            slot.ready_event.record(slot.copy_stream)

    def _decode_packed12_gpu(self, packed: np.ndarray, *, sample_count: int):
        """Compatibility decoder used by unit tests; returns uint16 on CUDA."""
        cp = self.cp
        host = np.asarray(packed, dtype=np.uint8)
        if host.ndim != 1:
            raise ValueError("packed-12 输入必须是一维 uint8 数组。")
        if sample_count < 0:
            raise ValueError("sample_count 必须为非负整数。")
        expected_bytes = (sample_count * 12 + 7) // 8
        if host.size != expected_bytes:
            raise ValueError(
                f"{sample_count} 个 packed-12 样本应为 {expected_bytes} 字节，"
                f"实际为 {host.size} 字节。"
            )
        packed_gpu = cp.asarray(host)
        pair_count, has_last_sample = divmod(sample_count, 2)
        decoded = cp.empty(sample_count, dtype=cp.uint16)
        if pair_count:
            triples = packed_gpu[: pair_count * 3].reshape(-1, 3)
            decoded[: pair_count * 2 : 2] = triples[:, 0].astype(cp.uint16) | (
                (triples[:, 1] & cp.uint8(0x0F)).astype(cp.uint16) << cp.uint16(8)
            )
            decoded[1 : pair_count * 2 : 2] = (
                (triples[:, 1] >> cp.uint8(4)).astype(cp.uint16)
                | (triples[:, 2].astype(cp.uint16) << cp.uint16(4))
            )
        if has_last_sample:
            tail = packed_gpu[pair_count * 3 :]
            decoded[-1] = tail[0].astype(cp.uint16) | (
                (tail[1] & cp.uint8(0x0F)).astype(cp.uint16) << cp.uint16(8)
            )
        return decoded

    def _decode_preloaded_to_values(
        self,
        slot: PackedTransferSlot,
        *,
        aline_count: int,
    ):
        """Wait for async H2D, then decode directly to reusable float32 A-lines."""
        if self._values_buffer is None:
            raise RuntimeError("请先调用 prepare_streaming()。")
        if aline_count <= 0 or aline_count > self._max_alines:
            raise ValueError(
                f"aline_count 必须位于 1..{self._max_alines}，实际为 {aline_count}。"
            )
        sample_count = aline_count * self.depth
        if sample_count % 2:
            raise ValueError("reference-guided GPU 流水线要求采样总数为偶数。")
        expected_bytes = sample_count * 3 // 2
        if slot.valid_bytes != expected_bytes:
            raise ValueError(
                f"预载 packed 字节数不匹配：应为 {expected_bytes}，"
                f"实际为 {slot.valid_bytes}。"
            )
        self.compute_stream.wait_event(slot.ready_event)
        flat = self._values_buffer[:sample_count]
        pair_count = sample_count // 2
        threads = 256
        blocks = (pair_count + threads - 1) // threads
        self._decode_u12_f32(
            (blocks,),
            (threads,),
            (
                slot.device[:expected_bytes],
                flat,
                np.int64(pair_count),
            ),
            stream=self.compute_stream,
        )
        return flat.reshape(aline_count, self.depth)

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        if value <= 0:
            raise ValueError("median 列数必须为正数。")
        return 1 << (int(value) - 1).bit_length()

    def _row_median(self, values, *, slot: int = 0):
        """Asynchronous row median for a finite float32 matrix with <=512 columns."""
        cp = self.cp
        if values.ndim != 2:
            raise ValueError(f"row median 输入必须是二维数组，实际为 {values.shape}。")
        rows, cols = int(values.shape[0]), int(values.shape[1])
        if cols <= 0 or cols > 512:
            raise ValueError(f"row median 当前支持 1..512 列，实际为 {cols}。")
        if values.dtype != cp.float32:
            values = values.astype(cp.float32, copy=False)
        if self._median_buffer0 is None or self._median_buffer1 is None:
            raise RuntimeError("请先调用 prepare_streaming() 分配 median workspace。")
        if rows > self._max_alines:
            raise ValueError(f"row median 行数 {rows} 超过 workspace {self._max_alines}。")
        output = (self._median_buffer0 if slot == 0 else self._median_buffer1)[:rows]
        sort_size = self._next_power_of_two(cols)
        self._row_median_kernel(
            (rows,),
            (256,),
            (
                values,
                output,
                np.int32(rows),
                np.int32(cols),
                np.int64(values.strides[0] // values.itemsize),
                np.int64(values.strides[1] // values.itemsize),
                np.int32(sort_size),
            ),
            stream=self.compute_stream,
        )
        return output

    def _clean_old_method(self, values):
        """Historical median-baseline NCC + LS subtraction, fused after FFT."""
        cp = self.cp
        baselines = self._row_median(values, slot=0)
        centered = values - baselines[:, None]

        correlation, dot = _correlation_terms_gpu(
            centered,
            self.kernel,
            self.template_energy,
            cp,
            self.fftconvolve,
        )

        row_count = int(values.shape[0])
        centers = self._centers_buffer[:row_count]
        coefficients = self._coefficients_buffer[:row_count]
        matched_u8 = self._matched_buffer[:row_count]
        self._select_match_kernel(
            (row_count,),
            (256,),
            (
                correlation,
                dot,
                self.template_energy,
                centers,
                coefficients,
                matched_u8,
                np.int32(row_count),
                np.int32(self.depth),
                np.int64(correlation.strides[0] // correlation.itemsize),
                np.int64(dot.strides[0] // dot.itemsize),
                self.kernel_peak_host,
                np.float32(self.minimum_fitted_peak_adc),
                np.float32(self.correlation_threshold),
            ),
            stream=self.compute_stream,
        )

        total = row_count * self.depth
        threads = 256
        blocks = (total + threads - 1) // threads
        self._subtract_selected_kernel(
            (blocks,),
            (threads,),
            (
                values,
                self.kernel,
                np.int32(self.kernel.size),
                np.int32(self.half_width),
                centers,
                coefficients,
                matched_u8,
                np.int64(total),
                np.int32(self.depth),
            ),
            stream=self.compute_stream,
        )
        return values, matched_u8

    def _robust_signal_power_sum(self, values, *, iterations: int = 3):
        """Exact QC power with no final full-depth residual materialisation."""
        cp = self.cp
        positions = self.full_positions
        mask = cp.ones(values.shape, dtype=cp.bool_)
        slope = cp.zeros(values.shape[0], dtype=cp.float32)
        intercept = cp.zeros_like(slope)

        # The mask computed after the last iteration is unused by the CPU
        # reference too: final residual is based on the fit entering iteration 3.
        for iteration in range(iterations):
            weights = mask.astype(cp.float32)
            sum_w = cp.maximum(weights.sum(axis=1), cp.float32(2.0))
            sum_z = cp.sum(weights * positions[None, :], axis=1, dtype=cp.float32)
            sum_x = cp.sum(weights * values, axis=1, dtype=cp.float32)
            sum_zz = cp.sum(
                weights * positions[None, :] * positions[None, :],
                axis=1,
                dtype=cp.float32,
            )
            sum_zx = cp.sum(
                weights * positions[None, :] * values,
                axis=1,
                dtype=cp.float32,
            )
            denominator = cp.maximum(
                sum_w * sum_zz - sum_z * sum_z, cp.float32(1e-6)
            )
            slope = (sum_w * sum_zx - sum_z * sum_x) / denominator
            intercept = (sum_x - slope * sum_z) / sum_w
            if iteration == iterations - 1:
                break
            residual = values - (
                intercept[:, None] + slope[:, None] * positions[None, :]
            )
            center = self._row_median(residual, slot=0)
            absolute_deviation = cp.abs(residual - center[:, None])
            scale = cp.float32(1.4826) * self._row_median(
                absolute_deviation, slot=1
            )
            scale = cp.maximum(scale, cp.float32(1e-3))
            mask = cp.abs(residual - center[:, None]) <= (
                cp.float32(4.0) * scale[:, None]
            )

        signal = slice(*self.signal_window)
        signal_residual = values[:, signal] - (
            intercept[:, None] + slope[:, None] * self.signal_positions[None, :]
        )
        return cp.sum(
            signal_residual * signal_residual,
            dtype=cp.float64,
        )

    def _excess_rms_projection(self, values):
        """CUDA adaptive excess-RMS without a full-depth detrended temporary."""
        cp = self.cp
        left, right = self.noise_windows
        signal = self.signal_window
        positions = self.noise_positions
        noise_values = values[:, self.noise_indices]

        weights = cp.ones_like(noise_values, dtype=cp.float32)
        slope = cp.zeros(values.shape[0], dtype=cp.float32)
        intercept = cp.zeros_like(slope)
        scale = cp.ones_like(slope)
        for _ in range(2):
            sum_w = cp.maximum(weights.sum(axis=1), cp.float32(2.0))
            weighted_z = weights * positions[None, :]
            sum_z = weighted_z.sum(axis=1, dtype=cp.float32)
            sum_x = (weights * noise_values).sum(axis=1, dtype=cp.float32)
            sum_zz = (weighted_z * positions[None, :]).sum(
                axis=1, dtype=cp.float32
            )
            sum_zx = (weighted_z * noise_values).sum(axis=1, dtype=cp.float32)
            denom = cp.maximum(
                sum_w * sum_zz - sum_z * sum_z, cp.float32(1e-6)
            )
            slope = (sum_w * sum_zx - sum_z * sum_x) / denom
            intercept = (sum_x - slope * sum_z) / sum_w
            noise_residual = noise_values - (
                intercept[:, None] + slope[:, None] * positions[None, :]
            )
            center = self._row_median(noise_residual, slot=0)
            absolute_deviation = cp.abs(noise_residual - center[:, None])
            scale = cp.float32(1.4826) * self._row_median(
                absolute_deviation, slot=1
            )
            scale = cp.maximum(scale, cp.float32(1e-3))
            weights = (
                cp.abs(noise_residual) <= cp.float32(4.0) * scale[:, None]
            ).astype(cp.float32)

        scale_limit = cp.float32(4.0) * cp.maximum(scale, cp.float32(1e-3))
        left_values = values[:, left[0] : left[1]] - (
            intercept[:, None] + slope[:, None] * self.left_positions[None, :]
        )
        right_values = values[:, right[0] : right[1]] - (
            intercept[:, None] + slope[:, None] * self.right_positions[None, :]
        )
        left_residual = cp.clip(
            left_values, -scale_limit[:, None], scale_limit[:, None]
        )
        right_residual = cp.clip(
            right_values, -scale_limit[:, None], scale_limit[:, None]
        )
        left_power = cp.mean(left_residual * left_residual, axis=1, dtype=cp.float32)
        right_power = cp.mean(
            right_residual * right_residual, axis=1, dtype=cp.float32
        )
        predicted_power = left_power[:, None] + (
            right_power - left_power
        )[:, None] * self.interpolation[None, :]
        predicted_power = cp.maximum(predicted_power, cp.float32(0.0))
        expected_noise_power = cp.mean(predicted_power, axis=1, dtype=cp.float32)

        signal_values = values[:, signal[0] : signal[1]] - (
            intercept[:, None] + slope[:, None] * self.signal_positions[None, :]
        )
        signal_power = cp.mean(signal_values * signal_values, axis=1, dtype=cp.float32)
        excess = cp.maximum(signal_power - expected_noise_power, cp.float32(0.0))
        return cp.sqrt(excess)

    def _process_values(self, values) -> GpuChunkResult:
        cp = self.cp
        if values.ndim != 2 or int(values.shape[1]) != self.depth:
            raise ValueError(
                f"values 必须为 (A-line 数, {self.depth})，实际为 {values.shape}。"
            )
        if values.dtype != cp.float32:
            values = values.astype(cp.float32, copy=False)

        before_power = self._robust_signal_power_sum(values)
        cleaned, matched = self._clean_old_method(values)
        after_power = self._robust_signal_power_sum(cleaned)
        projection = self._excess_rms_projection(cleaned)
        matched_count = cp.sum(matched, dtype=cp.int64)
        stats_gpu = cp.stack(
            (
                matched_count.astype(cp.float64),
                before_power.astype(cp.float64),
                after_power.astype(cp.float64),
            )
        )

        # Enqueue both tiny-QC and projection D2H copies into persistent pinned
        # host buffers, then synchronize exactly once for this whole frame/chunk.
        row_count = int(values.shape[0])
        host_projection = self._host_projection[:row_count]
        projection.get(
            out=host_projection, stream=self.compute_stream, blocking=False
        )
        stats_gpu.get(
            out=self._host_stats, stream=self.compute_stream, blocking=False
        )
        self.compute_stream.synchronize()
        stats = self._host_stats
        return GpuChunkResult(
            projection=np.array(host_projection, copy=True),
            matched_count=int(round(float(stats[0]))),
            signal_power_before=float(stats[1]),
            signal_power_after=float(stats[2]),
        )

    def process_preloaded_slot(
        self,
        slot: PackedTransferSlot,
        *,
        aline_count: int,
    ) -> GpuChunkResult:
        """Process one already-enqueued ping-pong input slot on the compute stream."""
        with self.cp.cuda.Device(self.device_id):
            with self.compute_stream:
                values = self._decode_preloaded_to_values(
                    slot, aline_count=aline_count
                )
                return self._process_values(values)

    def process_chunk(self, raw: np.ndarray) -> GpuChunkResult:
        """Compatibility path for already-decoded CPU A-lines."""
        values_np = np.asarray(raw)
        if values_np.ndim != 2 or values_np.shape[1] != self.depth:
            raise ValueError(
                f"raw 必须为 (A-line 数, {self.depth})，实际为 {values_np.shape}。"
            )
        with self.cp.cuda.Device(self.device_id):
            with self.compute_stream:
                values = self.cp.asarray(values_np, dtype=self.cp.float32)
                return self._process_values(values)

    def process_packed_chunk(
        self,
        packed: np.ndarray,
        *,
        aline_count: int,
    ) -> GpuChunkResult:
        """Compatibility packed path used outside the streaming loader."""
        if not isinstance(aline_count, int) or isinstance(aline_count, bool) or aline_count <= 0:
            raise ValueError("aline_count 必须是正整数。")
        host = np.asarray(packed, dtype=np.uint8)
        expected = aline_count * self.depth * 3 // 2
        if host.ndim != 1 or host.size != expected:
            raise ValueError(
                f"packed 字节数不匹配：应为 {expected}，实际为 {host.size}。"
            )
        self.prepare_streaming(
            max_packed_bytes=host.size,
            max_alines=aline_count,
        )
        slot = self.transfer_slots[0]
        slot.host[: host.size] = host
        self.enqueue_h2d(slot, nbytes=host.size)
        return self.process_preloaded_slot(slot, aline_count=aline_count)
