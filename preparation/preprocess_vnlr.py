import csv
import math
import os
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from glob import glob
import multiprocessing
from multiprocessing import Manager, Pool
from pathlib import Path

import av
import cv2
import numpy as np
import psutil
import torch
from sklearn.model_selection import train_test_split
from torchcodec.decoders import VideoDecoder

from transforms import TextTransform

# ── Deployment paths ─────────────────────────────────────────────────────────
DATASET_DIR  = '/app/dataset'
OUTPUT_DIR   = '/app/vnlr'
LABELS_DIR   = '/app/labels'
DONE_LOG     = '/app/resume_state/preprocess_done.txt'
OOM_LOG      = '/app/resume_state/preprocess_oom_retry.txt'


# ── Hardware-adaptive config ──────────────────────────────────────────────────
@dataclass
class HWConfig:
    num_gpus: int
    workers_per_gpu: int
    decode_threads: int
    crop_workers: int
    save_threads: int
    prefetch_q: int
    crop_q_depth: int
    save_q_depth: int
    det_chunk: int
    fan_chunk: int
    retry_det_chunk: int
    retry_fan_chunk: int
    use_fp16: bool
    use_compile: bool
    use_nvenc: bool
    oom_max_attempts: int
    oom_backoff_sec: float


def detect_hardware() -> HWConfig:
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs found.")

    vram_gb = min(
        torch.cuda.get_device_properties(i).total_memory / 1e9
        for i in range(n_gpus)
    )
    total_cores = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / 1e9

    # 2 workers/GPU when VRAM >= 12 GB: GPU is idle ~90% of the time waiting for
    # decode; a second worker hides that latency with compute overlap.
    workers_per_gpu = 2 if vram_gb >= 12.0 else 1
    total_workers = n_gpus * workers_per_gpu
    cores_per_worker = max(1, total_cores // total_workers)

    if cores_per_worker >= 18:
        decode_threads, crop_workers, save_threads = 6, 4, 3
    elif cores_per_worker >= 9:
        decode_threads, crop_workers, save_threads = 4, 3, 2
    elif cores_per_worker >= 5:
        decode_threads, crop_workers, save_threads = 3, 2, 2
    else:
        decode_threads, crop_workers, save_threads = 3, 2, 1

    # VRAM budget per worker (subtract ~0.8 GB for model weights + CUDA runtime)
    vram_budget = (vram_gb / workers_per_gpu) - 0.8
    bytes_per_det_frame = 1280 * 720 * 3 * 2  # fp16, 720p upper bound
    raw_det = int(vram_budget * 0.50 * 1e9 / bytes_per_det_frame)
    det_chunk = min(512, max(16, 2 ** math.floor(math.log2(max(raw_det, 16)))))

    bytes_per_fan_face = (3 * 256 * 256 + 68 * 64 * 64) * 4  # float32 input + heatmaps
    raw_fan = int(vram_budget * 0.35 * 1e9 / bytes_per_fan_face)
    fan_chunk = min(1024, max(16, 2 ** math.floor(math.log2(max(raw_fan, 16)))))

    retry_det_chunk = max(16, det_chunk // 8)
    retry_fan_chunk = max(16, fan_chunk // 8)

    # Queue depths from RAM — each queued video ≈ 400 MB
    per_worker_queue_gb = (ram_gb * 0.40) / total_workers
    prefetch_q   = max(8,  min(32, int(per_worker_queue_gb * 1e9 / (400 * 1e6))))
    crop_q_depth = max(4,  prefetch_q // 2)
    save_q_depth = max(16, prefetch_q * 2)

    # Enable NVENC on Turing+ (sm_75+); RTX 3060 = Ampere sm_86
    use_nvenc = any(
        torch.cuda.get_device_capability(i)[0] >= 7
        for i in range(n_gpus)
    )
    # torch.compile causes CUDA Graph memory fragmentation when workers share a GPU
    use_compile = (workers_per_gpu == 1 and n_gpus == 1)

    return HWConfig(
        num_gpus=n_gpus, workers_per_gpu=workers_per_gpu,
        decode_threads=decode_threads, crop_workers=crop_workers, save_threads=save_threads,
        prefetch_q=prefetch_q, crop_q_depth=crop_q_depth, save_q_depth=save_q_depth,
        det_chunk=det_chunk, fan_chunk=fan_chunk,
        retry_det_chunk=retry_det_chunk, retry_fan_chunk=retry_fan_chunk,
        use_fp16=True, use_compile=use_compile, use_nvenc=use_nvenc,
        oom_max_attempts=3, oom_backoff_sec=1.5,
    )


_CFG = detect_hardware()
# ─────────────────────────────────────────────────────────────────────────────


def load_video(path):
    return VideoDecoder(path).get_all_frames().data.permute(0, 2, 3, 1).numpy()


# Tested once at startup; workers read this flag (set in __main__)
_NVENC_OK: bool = True


def _write_video_pyav(filename, vid, fps, codec, options):
    """Write via PyAV (no subprocess spawn overhead, ~1.3x faster than ffmpeg pipe)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    h, w = vid.shape[1], vid.shape[2]
    container = av.open(filename, 'w')
    stream = container.add_stream(codec, rate=fps)
    stream.width, stream.height = w, h
    stream.pix_fmt = 'yuv420p'
    stream.options = options
    for frame_np in vid:
        frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def save2vid(filename, vid, fps):
    """Try NVENC first (GPU 0 — CUDA_VISIBLE_DEVICES already virtualises the right GPU),
    fall back to libx264 ultrafast on failure. Both paths use PyAV — no subprocess spawn."""
    if _NVENC_OK:
        try:
            # '-gpu 0' always: each worker has CUDA_VISIBLE_DEVICES set so its GPU IS index 0
            _write_video_pyav(filename, vid, fps,
                              'h264_nvenc', {'preset': 'p1', 'gpu': '0'})
            return
        except Exception:
            pass
    _write_video_pyav(filename, vid, fps,
                      'libx264', {'preset': 'ultrafast', 'crf': '23'})


_SENTINEL = object()


def _infer_with_oom_retry(fast_pipeline, frames,
                          max_attempts=None, backoff_sec=None):
    """Run fast_pipeline(frames) with adaptive OOM retry.

    Strategy on torch.cuda.OutOfMemoryError:
      1. empty_cache + synchronize to release any partial allocations.
      2. Sleep `backoff_sec * (attempt+1)` so the sibling worker on the same
         GPU has time to drain its own peak (when 2 workers/GPU contend, the
         OOM is usually transient — peaks rarely overlap for long).
      3. Halve det_chunk and fan_chunk for the next attempt (mutating the
         pipeline in place; restored in finally).

    Raises torch.cuda.OutOfMemoryError if all attempts fail — caller should
    defer the file to the OOM retry log for a single-worker pass at the end.
    """
    if max_attempts is None:
        max_attempts = _CFG.oom_max_attempts
    if backoff_sec is None:
        backoff_sec = _CFG.oom_backoff_sec

    original_det = fast_pipeline.det_chunk
    original_fan = fast_pipeline.fan_chunk
    last_exc = None
    try:
        for attempt in range(max_attempts):
            try:
                return fast_pipeline(frames)
            except torch.cuda.OutOfMemoryError as e:
                last_exc = e
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass
                if attempt < max_attempts - 1:
                    time.sleep(backoff_sec * (attempt + 1))
                    fast_pipeline.det_chunk = max(8, fast_pipeline.det_chunk // 2)
                    fast_pipeline.fan_chunk = max(8, fast_pipeline.fan_chunk // 2)
        raise last_exc
    finally:
        fast_pipeline.det_chunk = original_det
        fast_pipeline.fan_chunk = original_fan


def worker(worker_id, gpu_id, file_chunk, train_set, val_set, test_set,
           done_set, lock, counters, start_time, retry_mode=False):
    # ── CPU pinning ─────────────────────────────────────────────────────────
    total_workers   = _CFG.num_gpus * _CFG.workers_per_gpu
    total_cores     = psutil.cpu_count(logical=True)
    cores_per_worker = max(1, total_cores // total_workers)
    cpu_start = worker_id * cores_per_worker
    try:
        psutil.Process().cpu_affinity(
            list(range(cpu_start, cpu_start + cores_per_worker))
        )
    except Exception:
        pass

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # PyTorch performance flags
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    from detectors.retinaface.detector import LandmarksDetector
    from detectors.retinaface.video_process import VideoProcess
    from fast_pipeline import BatchedLandmarkPipeline

    landmarks_obj = LandmarksDetector(device='cuda:0')
    video_process = VideoProcess(convert_gray=False)
    tokenizer     = TextTransform()

    det_chunk = _CFG.retry_det_chunk if retry_mode else _CFG.det_chunk
    fan_chunk = _CFG.retry_fan_chunk if retry_mode else _CFG.fan_chunk

    fast_pipeline = BatchedLandmarkPipeline(
        landmarks_obj.face_detector,
        landmarks_obj.landmark_detector,
        device='cuda:0',
        det_chunk=det_chunk,
        fan_chunk=fan_chunk,
        fp16=_CFG.use_fp16,
        use_compile=_CFG.use_compile,
    )

    log_suffix = f"w{worker_id}.retry" if retry_mode else f"w{worker_id}"
    train_csv = open(f"{LABELS_DIR}/train_{log_suffix}.csv", "a")
    val_csv   = open(f"{LABELS_DIR}/val_{log_suffix}.csv",   "a")
    test_csv  = open(f"{LABELS_DIR}/test_{log_suffix}.csv",  "a")
    done_log  = open(f"{DONE_LOG}.{log_suffix}", "a")
    # OOM-deferred log: ở pass chính, các video không retry được sẽ vào đây để
    # pass cuối (single-worker) chạy lại. Ở pass retry, file này KHÔNG được mở
    # tránh đệ quy vô hạn — OOM ở pass retry sẽ ghi vào done_log như INFER-ERR.
    oom_log   = open(f"{OOM_LOG}.{log_suffix}", "a") if not retry_mode else None

    total_files  = counters['total']
    initial_done = counters['initial_done']

    # ── Stage 1: Decode  ───────────────────────────────────────────────────
    decode_q = queue.Queue(maxsize=_CFG.prefetch_q)

    def decode_one(f):
        name = Path(f).name
        if name in done_set:
            decode_q.put((f, 'resume', None))
            return
        try:
            decode_q.put((f, 'ok', load_video(f)))
        except Exception:
            decode_q.put((f, 'error', None))

    def decode_producer():
        with ThreadPoolExecutor(max_workers=_CFG.decode_threads) as pool:
            for fut in [pool.submit(decode_one, f) for f in file_chunk]:
                fut.result()

    decode_thread = threading.Thread(target=decode_producer, daemon=True)
    decode_thread.start()

    # ── Stage 3: Crop  ────────────────────────────────────────────────────
    # KEY: GPU inference puts (frames, landmarks, meta) here;
    # crop workers pick it up so GPU can immediately start the next video.
    crop_q = queue.Queue(maxsize=_CFG.crop_q_depth)

    # ── Stage 4: Save  ────────────────────────────────────────────────────
    save_q = queue.Queue(maxsize=_CFG.save_q_depth)

    def crop_consumer():
        while True:
            item = crop_q.get()
            if item is _SENTINEL:
                break
            f         = item['f']
            frames    = item['frames']
            frames_gpu = item['frames_gpu']
            landmarks = item['landmarks']
            idx       = item['idx']
            name = Path(f).name

            elapsed = time.time() - start_time
            ts = time.strftime('%H:%M:%S', time.gmtime(elapsed))

            try:
                video_data = video_process(frames, landmarks, frames_gpu=frames_gpu)
            except (OverflowError, TypeError, AssertionError, UnboundLocalError):
                print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - CROP-ERR    {name}", flush=True)
                done_log.write(name + '\n'); done_log.flush()
                continue

            if video_data is None:
                print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - NO-FACE     {name}", flush=True)
                done_log.write(name + '\n'); done_log.flush()
                continue

            csv_path = f.replace('.mp4', '.csv')
            try:
                transcript = []
                with open(csv_path) as csvfile:
                    for row in csv.DictReader(csvfile):
                        transcript.append(row['Word'].strip().rstrip('.,!?\'"').lower())
            except FileNotFoundError:
                done_log.write(name + '\n'); done_log.flush()
                continue

            text = ' '.join(transcript)
            ids  = ' '.join(str(x) for x in tokenizer.tokenize(text).tolist())
            save_q.put({
                'f': f, 'name': name,
                'video_tensor': torch.tensor(video_data),
                'text': text, 'ids': ids, 'idx': idx,
            })

    def save_consumer():
        while True:
            item = save_q.get()
            if item is _SENTINEL:
                break
            f, name = item['f'], item['name']
            video_tensor, text, ids, idx = (
                item['video_tensor'], item['text'], item['ids'], item['idx'])

            out_vid = f"{OUTPUT_DIR}/video/{name}"
            out_txt = f"{OUTPUT_DIR}/text/{Path(f).stem}.txt"
            vid_np  = video_tensor.numpy()

            save2vid(out_vid, vid_np, 25)

            os.makedirs(os.path.dirname(out_txt), exist_ok=True)
            with open(out_txt, 'w') as ft:
                ft.write(text)

            speaker = Path(f).stem.rsplit('_', 1)[0]
            row     = f"vnlr,video/{name},{len(video_tensor)},{ids}\n"
            if speaker in train_set:
                train_csv.write(row); train_csv.flush()
            elif speaker in val_set:
                val_csv.write(row);   val_csv.flush()
            elif speaker in test_set:
                test_csv.write(row);  test_csv.flush()

            done_log.write(name + '\n'); done_log.flush()
            elapsed = time.time() - start_time
            ts = time.strftime('%H:%M:%S', time.gmtime(elapsed))
            print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - OK   {name}", flush=True)

    crop_threads = [threading.Thread(target=crop_consumer, daemon=True)
                    for _ in range(_CFG.crop_workers)]
    save_threads = [threading.Thread(target=save_consumer, daemon=True)
                    for _ in range(_CFG.save_threads)]
    for t in crop_threads + save_threads:
        t.start()

    # ── Stage 2: GPU inference loop (main worker thread) ──────────────────
    # This thread ONLY does GPU inference and immediately queues result for
    # crop workers — it never waits for CPU crop to finish.
    try:
        pending = len(file_chunk)
        while pending > 0:
            f, status, frames = decode_q.get()
            pending -= 1
            name = Path(f).name

            with lock:
                counters['done'] += 1
                idx = counters['done']

            elapsed = time.time() - start_time
            ts = time.strftime('%H:%M:%S', time.gmtime(elapsed))

            if status == 'resume':
                if (idx - initial_done) <= 5 or (idx % 500 == 0):
                    print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - RESUME-SKIP {name}", flush=True)
                continue

            if status == 'error':
                print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - DECODE-ERR  {name}", flush=True)
                done_log.write(name + '\n'); done_log.flush()
                continue

            try:
                landmarks, frames_gpu = _infer_with_oom_retry(fast_pipeline, frames)
            except torch.cuda.OutOfMemoryError:
                # Đã thử OOM_MAX_ATTEMPTS lần với chunk giảm dần mà vẫn OOM.
                # Defer sang retry pass (1-worker) thay vì đánh dấu done.
                if oom_log is not None:
                    tag = "OOM-DEFER "
                    oom_log.write(f + '\n'); oom_log.flush()
                else:
                    # Đang ở retry pass — không có tầng dự phòng tiếp theo, bỏ luôn.
                    tag = "OOM-FINAL "
                    done_log.write(name + '\n'); done_log.flush()
                print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - {tag} {name}", flush=True)
                # Giải phóng tham chiếu frames (RAM lớn) và GPU cache trước khi tiếp tục
                del frames
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            except Exception as _infer_exc:
                import traceback as _tb
                print(f"[{ts}] {idx}/{total_files} W{worker_id}/G{gpu_id} - INFER-ERR   {name} | {type(_infer_exc).__name__}: {_infer_exc!s:.200}", flush=True)
                _tb.print_exc()
                done_log.write(name + '\n'); done_log.flush()
                continue

            # Hand off immediately — GPU loop does NOT wait for crop
            # frames_gpu reuses the tensor already on-device from inference (no second upload)
            crop_q.put({'f': f, 'frames': frames, 'frames_gpu': frames_gpu,
                        'landmarks': landmarks, 'idx': idx})

    finally:
        decode_thread.join()
        for _ in crop_threads:
            crop_q.put(_SENTINEL)
        for t in crop_threads:
            t.join()
        for _ in save_threads:
            save_q.put(_SENTINEL)
        for t in save_threads:
            t.join()
        train_csv.close(); val_csv.close(); test_csv.close(); done_log.close()
        if oom_log is not None:
            oom_log.close()


def _shard_suffixes(num_workers):
    """All shard suffixes a worker may have written (normal + retry pass)."""
    suffixes = [f"w{w}" for w in range(num_workers)]
    suffixes += [f"w{w}.retry" for w in range(num_workers)]
    return suffixes


def merge_shards(num_workers):
    """Merge per-worker CSV + done-log shards into the canonical files.
    Bao gồm cả shard từ retry pass (suffix '.retry').
    """
    suffixes = _shard_suffixes(num_workers)
    for split in ('train', 'val', 'test'):
        with open(f"{LABELS_DIR}/{split}.csv", "a") as out:
            for sfx in suffixes:
                shard = f"{LABELS_DIR}/{split}_{sfx}.csv"
                if os.path.exists(shard):
                    with open(shard) as inp:
                        out.write(inp.read())
                    os.remove(shard)
    with open(DONE_LOG, "a") as out:
        for sfx in suffixes:
            shard = f"{DONE_LOG}.{sfx}"
            if os.path.exists(shard):
                with open(shard) as inp:
                    out.write(inp.read())
                os.remove(shard)


def collect_oom_files(num_workers):
    """Read & remove per-worker OOM shards. Returns list of full file paths."""
    files = []
    for w in range(num_workers):
        shard = f"{OOM_LOG}.w{w}"
        if os.path.exists(shard):
            with open(shard) as inp:
                files.extend(line.strip() for line in inp if line.strip())
            os.remove(shard)
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f); unique.append(f)
    return unique


def load_done_set():
    """Aggressively load every possible done-log location so resume always works."""
    done_set = set()
    paths_checked = []

    def absorb(path):
        paths_checked.append(path)
        if os.path.exists(path):
            with open(path) as f:
                added = set(line.strip() for line in f if line.strip())
            done_set.update(added)
            return len(added)
        return 0

    main_count = absorb(DONE_LOG)
    print(f"  main done log [{DONE_LOG}]: {main_count} entries")

    # Absorb leftover shards from previous runs (gpu-style, w-style, and retry-style)
    leftover = 0
    for w in range(_CFG.num_gpus * _CFG.workers_per_gpu + _CFG.num_gpus):  # cover both schemes
        for pat in (f"{DONE_LOG}.w{w}",
                    f"{DONE_LOG}.w{w}.retry",
                    f"{DONE_LOG}.gpu{w}"):
            if os.path.exists(pat):
                leftover += absorb(pat)
    if leftover:
        print(f"  + {leftover} entries from leftover shards")

    return done_set


if __name__ == '__main__':
    # 'spawn' creates clean child processes with no inherited CUDA state.
    # 'fork' (Linux default) copies the parent's file descriptors and can
    # corrupt the CUDA driver context that torch initialises at import time.
    multiprocessing.set_start_method('spawn', force=True)
    start = time.time()

    print("=" * 60)
    print("Hardware config (auto-detected):")
    print(f"  GPUs: {_CFG.num_gpus}  workers/GPU: {_CFG.workers_per_gpu}"
          f"  total workers: {_CFG.num_gpus * _CFG.workers_per_gpu}")
    print(f"  Threads/worker: decode={_CFG.decode_threads}"
          f"  crop={_CFG.crop_workers}  save={_CFG.save_threads}")
    print(f"  Batch sizes: DET={_CFG.det_chunk}  FAN={_CFG.fan_chunk}")
    print(f"  Retry batch: DET={_CFG.retry_det_chunk}  FAN={_CFG.retry_fan_chunk}")
    print(f"  Queues: prefetch={_CFG.prefetch_q}"
          f"  crop={_CFG.crop_q_depth}  save={_CFG.save_q_depth}")
    print(f"  FP16={_CFG.use_fp16}  compile={_CFG.use_compile}  nvenc={_CFG.use_nvenc}")
    print("=" * 60)

    # Create all required directories upfront
    for dirpath in [
        DATASET_DIR,
        f"{OUTPUT_DIR}/video",
        f"{OUTPUT_DIR}/text",
        LABELS_DIR,
        str(Path(DONE_LOG).parent),
    ]:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
            print(f"Created directory: {dirpath}")

    # Probe NVENC once so workers know whether to attempt it
    if _CFG.use_nvenc:
        import numpy as _np
        _probe = _np.zeros((4, 256, 256, 3), dtype=_np.uint8)
        try:
            _write_video_pyav('/tmp/_nvenc_probe.mp4', _probe, 25,
                              'h264_nvenc', {'preset': 'p1', 'gpu': '0'})
            print("NVENC: available — using hardware encoding")
        except Exception as _e:
            _NVENC_OK = False
            print(f"NVENC: unavailable ({_e}) — using libx264 ultrafast")
        del _probe
    else:
        _NVENC_OK = False
        print("NVENC: disabled by hardware config — using libx264 ultrafast")

    files = sorted(glob(f'{DATASET_DIR}/*.mp4'))
    print(f"Found {len(files)} files in {DATASET_DIR}")

    print("\nLoading resume state:")
    done_set = load_done_set()
    print(f"  → {len(done_set)} files already processed (will be skipped)\n")

    # Speaker split
    speakers = sorted(set(Path(f).stem.rsplit('_', 1)[0] for f in files))
    train_sp, temp  = train_test_split(speakers, test_size=0.2, random_state=42)
    val_sp, test_sp = train_test_split(temp,     test_size=0.5, random_state=42)
    train_set, val_set, test_set = set(train_sp), set(val_sp), set(test_sp)

    actual_gpus   = _CFG.num_gpus  # detect_hardware() already called device_count()
    total_workers = actual_gpus * _CFG.workers_per_gpu
    print(f"Launching {total_workers} workers ({actual_gpus} GPUs × {_CFG.workers_per_gpu} workers/GPU)")

    # Round-robin shard so each worker gets evenly-mixed file IDs
    chunks = [files[i::total_workers] for i in range(total_workers)]
    print(f"Files per worker: min={min(len(c) for c in chunks)} max={max(len(c) for c in chunks)}\n")

    with Manager() as manager:
        lock     = manager.Lock()
        counters = manager.dict(
            done=len(done_set),          # already-completed files count toward progress
            total=len(files),
            initial_done=len(done_set),  # snapshot for filtering log spam
        )

        args = []
        for wid in range(total_workers):
            gpu_id = wid % actual_gpus  # round-robin GPU assignment
            args.append((
                wid, gpu_id, chunks[wid],
                train_set, val_set, test_set,
                done_set, lock, counters, start,
                False,  # retry_mode
            ))

        with Pool(processes=total_workers) as pool:
            pool.starmap(worker, args)

    # ── OOM retry pass: 1 worker, full VRAM, chunk size rất nhỏ ──────────
    oom_files = collect_oom_files(total_workers)
    if oom_files:
        print(f"\n{'='*60}")
        print(f"OOM retry pass: {len(oom_files)} files deferred from main pass")
        print(f"  → spawning 1 worker with DET_CHUNK={_CFG.retry_det_chunk}, "
              f"FAN_CHUNK={_CFG.retry_fan_chunk} on GPU 0")
        print(f"{'='*60}\n")

        # Reload done_set: main pass đã ghi nhiều file mới
        done_set_retry = load_done_set()

        with Manager() as manager:
            lock_r = manager.Lock()
            counters_r = manager.dict(
                done=0,
                total=len(oom_files),
                initial_done=0,
            )
            retry_args = [(
                0, 0, oom_files,
                train_set, val_set, test_set,
                done_set_retry, lock_r, counters_r, time.time(),
                True,  # retry_mode
            )]
            with Pool(processes=1) as pool:
                pool.starmap(worker, retry_args)
    else:
        print("\nNo OOM-deferred files — retry pass skipped.")

    merge_shards(total_workers)

    elapsed = time.time() - start
    print(f"\nDone in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"Total files: {len(files)}")
    if oom_files:
        print(f"OOM-retry pass processed: {len(oom_files)} files")
