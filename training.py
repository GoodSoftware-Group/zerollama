#!/usr/bin/env python3
"""
Persistent Training Worker for ZoeyAI Enterprise

This worker runs as a long-lived process, accepting training jobs via Unix socket.
Benefits:
- Model and tokenizer loaded once, reused across iterations
- PyTorch/Transformers stay in memory
- GPU memory stays allocated (faster subsequent training)
- Reduced startup overhead from ~30s to <1s per iteration
- Job queue for handling multiple concurrent clients

Protocol (JSON over Unix socket):
- Each message is a JSON object followed by newline
- Request: {"cmd": "train", "data": {...}} or {"cmd": "ping"} or {"cmd": "shutdown"}
- Request: {"cmd": "run_script", "data": {"script_path": "/path/to/train_script.py", ...}}
- Request: {"cmd": "submit_job", "data": {...}} - Queue a job, returns job_id
- Request: {"cmd": "job_status", "job_id": "..."} - Get job status
- Request: {"cmd": "list_jobs"} - List all jobs
- Request: {"cmd": "cancel_job", "job_id": "..."} - Cancel a pending job
- Response: {"status": "ok/error", "progress": 0-100, ...}
"""

import os
import sys
import json
import socket
import signal
import traceback
import subprocess
import re
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import threading
import time
from queue import Queue, Empty
from datetime import datetime

# Training imports (loaded once)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Optional: bitsandbytes for QLoRA
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


# ============================================================================
# Job Queue System
# ============================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    cmd: str  # "train" or "run_script"
    data: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    progress_message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    submitted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    client_id: Optional[str] = None  # Track which client submitted
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "cmd": self.cmd,
            "status": self.status.value,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "result": self.result,
            "error": self.error,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class JobQueue:
    """Thread-safe job queue with status tracking"""
    
    def __init__(self, max_history: int = 100):
        self._queue: deque[Job] = deque()
        self._jobs: Dict[str, Job] = {}  # All jobs by ID
        self._lock = threading.RLock()
        self._current_job: Optional[Job] = None
        self._max_history = max_history
        self._job_available = threading.Event()
        
    def submit(self, cmd: str, data: Dict[str, Any], client_id: Optional[str] = None) -> str:
        """Submit a job to the queue, returns job_id"""
        with self._lock:
            job_id = str(uuid.uuid4())[:8]  # Short ID for readability
            job = Job(
                id=job_id,
                cmd=cmd,
                data=data,
                client_id=client_id,
            )
            self._queue.append(job)
            self._jobs[job_id] = job
            self._job_available.set()
            self._cleanup_old_jobs()
            print(f"WORKER: Job {job_id} queued ({cmd}), queue size: {len(self._queue)}", flush=True)
            return job_id
    
    def get_next(self, timeout: float = 1.0) -> Optional[Job]:
        """Get next job from queue (blocks until available or timeout)"""
        if self._job_available.wait(timeout=timeout):
            with self._lock:
                if self._queue:
                    job = self._queue.popleft()
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.utcnow().isoformat()
                    self._current_job = job
                    if not self._queue:
                        self._job_available.clear()
                    return job
        return None
    
    def complete_job(self, job_id: str, result: Dict[str, Any]):
        """Mark job as completed with result"""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = JobStatus.COMPLETED
                job.result = result
                job.progress = 100.0
                job.completed_at = datetime.utcnow().isoformat()
                if self._current_job and self._current_job.id == job_id:
                    self._current_job = None
                print(f"WORKER: Job {job_id} completed", flush=True)
    
    def fail_job(self, job_id: str, error: str):
        """Mark job as failed with error"""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = JobStatus.FAILED
                job.error = error
                job.completed_at = datetime.utcnow().isoformat()
                if self._current_job and self._current_job.id == job_id:
                    self._current_job = None
                print(f"WORKER: Job {job_id} failed: {error}", flush=True)
    
    def update_progress(self, job_id: str, progress: float, message: str = ""):
        """Update job progress"""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.progress = progress
                job.progress_message = message
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job (cannot cancel running jobs)"""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if job.status == JobStatus.PENDING:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.utcnow().isoformat()
                    # Remove from queue
                    self._queue = deque(j for j in self._queue if j.id != job_id)
                    if not self._queue:
                        self._job_available.clear()
                    print(f"WORKER: Job {job_id} cancelled", flush=True)
                    return True
        return False
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        with self._lock:
            return self._jobs.get(job_id)
    
    def get_current_job(self) -> Optional[Job]:
        """Get currently running job"""
        with self._lock:
            return self._current_job
    
    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent jobs"""
        with self._lock:
            jobs = list(self._jobs.values())
            # Sort by submitted_at descending
            jobs.sort(key=lambda j: j.submitted_at, reverse=True)
            return [j.to_dict() for j in jobs[:limit]]
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._lock:
            pending = sum(1 for j in self._jobs.values() if j.status == JobStatus.PENDING)
            running = sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)
            completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
            failed = sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)
            return {
                "queue_length": len(self._queue),
                "pending": pending,
                "running": running,
                "completed": completed,
                "failed": failed,
                "current_job_id": self._current_job.id if self._current_job else None,
            }
    
    def _cleanup_old_jobs(self):
        """Remove old completed/failed jobs to limit memory"""
        with self._lock:
            if len(self._jobs) > self._max_history * 2:
                # Keep only recent jobs
                jobs = list(self._jobs.values())
                jobs.sort(key=lambda j: j.submitted_at, reverse=True)
                # Keep pending, running, and recent completed
                keep_jobs = {}
                for job in jobs:
                    if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                        keep_jobs[job.id] = job
                    elif len(keep_jobs) < self._max_history:
                        keep_jobs[job.id] = job
                self._jobs = keep_jobs


# Global job queue
JOB_QUEUE = JobQueue()


# ============================================================================
# Worker State
# ============================================================================

# Global state (persisted across requests)
class WorkerState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.socket_path: Optional[str] = None
        self.running = True
        self.client_socket = None  # For sending progress updates (legacy)
        self.current_job_id: Optional[str] = None  # Current job being processed
        self._clients: Dict[str, socket.socket] = {}  # Connected clients by ID
        self._clients_lock = threading.Lock()
        
    def register_client(self, client_id: str, sock: socket.socket):
        """Register a connected client"""
        with self._clients_lock:
            self._clients[client_id] = sock
            
    def unregister_client(self, client_id: str):
        """Unregister a client"""
        with self._clients_lock:
            self._clients.pop(client_id, None)
            
    def get_client_socket(self, client_id: str) -> Optional[socket.socket]:
        """Get socket for a client"""
        with self._clients_lock:
            return self._clients.get(client_id)
    
    def broadcast_to_job_owner(self, job: Job, message: Dict[str, Any]):
        """Send message to the client that submitted the job"""
        if job.client_id:
            sock = self.get_client_socket(job.client_id)
            if sock:
                try:
                    sock.sendall((json.dumps(message) + "\n").encode())
                except:
                    pass  # Client may have disconnected
        
    def load_model(self, model_name: str, use_lora: bool = True, use_qlora: bool = False):
        """Load model and tokenizer (only if different from current)"""
        if self.current_model_name == model_name and self.model is not None:
            print(f"WORKER: Model {model_name} already loaded, reusing", flush=True)
            return True
            
        print(f"WORKER: Loading model {model_name}...", flush=True)
        
        try:
            # Cleanup previous model
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Quantization config for QLoRA
            bnb_config = None
            if use_qlora and HAS_BNB:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                quantization_config=bnb_config,
            )
            
            # Prepare for LoRA if needed
            if use_lora or use_qlora:
                if use_qlora:
                    self.model = prepare_model_for_kbit_training(self.model)
                    
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
                
            self.current_model_name = model_name
            print(f"WORKER: Model {model_name} loaded successfully on {self.device}", flush=True)
            return True
            
        except Exception as e:
            print(f"WORKER ERROR: Failed to load model: {e}", flush=True)
            traceback.print_exc()
            return False
            
    def send_progress(self, progress: float, message: str = ""):
        """Send progress update to client (legacy) and update job queue"""
        # Update job queue progress
        if self.current_job_id:
            JOB_QUEUE.update_progress(self.current_job_id, progress, message)
            # Send to job owner
            job = JOB_QUEUE.get_job(self.current_job_id)
            if job:
                self.broadcast_to_job_owner(job, {
                    "type": "progress",
                    "job_id": self.current_job_id,
                    "progress": progress,
                    "message": message
                })
        
        # Legacy: direct socket (for backward compatibility)
        if self.client_socket:
            try:
                response = {
                    "type": "progress",
                    "progress": progress,
                    "message": message
                }
                self.client_socket.sendall((json.dumps(response) + "\n").encode())
            except:
                pass  # Client may have disconnected

STATE = WorkerState()


# ============================================================================
# Job Processor Thread
# ============================================================================

def job_processor():
    """Background thread that processes jobs from the queue"""
    print("WORKER: Job processor thread started", flush=True)
    
    while STATE.running:
        try:
            job = JOB_QUEUE.get_next(timeout=1.0)
            if job is None:
                continue
                
            print(f"WORKER: Processing job {job.id} ({job.cmd})", flush=True)
            STATE.current_job_id = job.id
            
            try:
                if job.cmd == "train":
                    result = process_training_request(job.data)
                elif job.cmd == "run_script":
                    result = run_local_script(job.data)
                else:
                    result = {"status": "error", "error": f"Unknown job command: {job.cmd}"}
                
                if result.get("status") == "ok":
                    JOB_QUEUE.complete_job(job.id, result)
                    # Notify client
                    STATE.broadcast_to_job_owner(job, {
                        "type": "job_completed",
                        "job_id": job.id,
                        "result": result
                    })
                else:
                    JOB_QUEUE.fail_job(job.id, result.get("error", "Unknown error"))
                    STATE.broadcast_to_job_owner(job, {
                        "type": "job_failed",
                        "job_id": job.id,
                        "error": result.get("error", "Unknown error")
                    })
                    
            except Exception as e:
                traceback.print_exc()
                JOB_QUEUE.fail_job(job.id, str(e))
                STATE.broadcast_to_job_owner(job, {
                    "type": "job_failed",
                    "job_id": job.id,
                    "error": str(e)
                })
            finally:
                STATE.current_job_id = None
                
        except Exception as e:
            print(f"WORKER: Job processor error: {e}", flush=True)
            traceback.print_exc()
            time.sleep(1)
    
    print("WORKER: Job processor thread stopped", flush=True)


def process_training_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Process a training request"""
    try:
        # Extract parameters
        model_name = request.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")
        output_dir = request.get("output_dir", "/tmp/training_output")
        training_data = request.get("training_data", [])  # List of {"prompt": ..., "response": ...}
        num_epochs = request.get("num_epochs", 3)
        batch_size = request.get("batch_size", 4)
        learning_rate = request.get("learning_rate", 2e-4)
        use_lora = request.get("use_lora", True)
        use_qlora = request.get("use_qlora", False)
        lora_rank = request.get("lora_rank", 16)
        lora_alpha = request.get("lora_alpha", 32.0)
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load model (reuses if same model)
        STATE.send_progress(5.0, "Loading model...")
        if not STATE.load_model(model_name, use_lora, use_qlora):
            return {"status": "error", "error": "Failed to load model"}
            
        STATE.send_progress(20.0, "Preparing dataset...")
        
        # Prepare dataset
        def format_sample(sample):
            return f"### Instruction:\n{sample['prompt']}\n\n### Response:\n{sample['response']}"
            
        texts = [format_sample(s) for s in training_data]
        
        # Tokenize
        def tokenize_fn(examples):
            return STATE.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )
            
        dataset = Dataset.from_dict({"text": texts})
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        
        STATE.send_progress(30.0, "Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            bf16=STATE.device == "cuda",
            report_to="none",
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
        )
        
        # Custom callback for progress
        class ProgressCallback:
            def __init__(self, total_steps):
                self.total_steps = total_steps
                
            def on_step_end(self, args, state, control, **kwargs):
                if self.total_steps > 0:
                    progress = 30.0 + (state.global_step / self.total_steps) * 60.0
                    STATE.send_progress(progress, f"Training step {state.global_step}/{self.total_steps}")
                    
        # Calculate total steps
        total_steps = (len(tokenized) // batch_size) * num_epochs
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=STATE.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=STATE.model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )
        
        # Add progress callback
        progress_cb = ProgressCallback(total_steps)
        
        # Override step callback
        original_training_step = trainer.training_step
        step_count = [0]
        def training_step_with_progress(*args, **kwargs):
            result = original_training_step(*args, **kwargs)
            step_count[0] += 1
            if total_steps > 0:
                progress = 30.0 + (step_count[0] / total_steps) * 60.0
                if step_count[0] % 5 == 0:  # Update every 5 steps
                    STATE.send_progress(progress, f"Training step {step_count[0]}/{total_steps}")
            return result
        trainer.training_step = training_step_with_progress
        
        # Train
        train_result = trainer.train()
        
        STATE.send_progress(90.0, "Saving model...")
        
        # Save the LoRA adapter
        adapter_path = os.path.join(output_dir, "lora_adapter")
        STATE.model.save_pretrained(adapter_path)
        STATE.tokenizer.save_pretrained(adapter_path)
        
        STATE.send_progress(95.0, "Training complete")
        
        return {
            "status": "ok",
            "output_dir": output_dir,
            "adapter_path": adapter_path,
            "train_loss": train_result.training_loss,
            "train_samples": len(training_data),
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def run_local_script(request: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a local training script and monitor its progress.
    
    This allows the worker to control external training scripts (like the
    dynamically generated train_script.py from huggingface_backend.rs) while
    keeping the GPU/PyTorch environment persistent.
    
    Args:
        request: Dict containing:
            - script_path: Path to the Python script to execute
            - script_args: Optional list of command line arguments
            - working_dir: Optional working directory
            - env: Optional dict of environment variables to add
            - timeout: Optional timeout in seconds (default: 3600)
    
    Returns:
        Dict with status, output, and any parsed progress info
    """
    try:
        script_path = request.get("script_path")
        if not script_path:
            return {"status": "error", "error": "script_path is required"}
            
        script_path = Path(script_path)
        if not script_path.exists():
            return {"status": "error", "error": f"Script not found: {script_path}"}
            
        script_args = request.get("script_args", [])
        working_dir = request.get("working_dir")
        extra_env = request.get("env", {})
        timeout_secs = request.get("timeout", 3600)  # 1 hour default
        
        STATE.send_progress(0.0, f"Starting script: {script_path.name}")
        
        # Prepare environment (inherit current + add extras)
        env = os.environ.copy()
        env.update(extra_env)
        
        # Build command
        cmd = [sys.executable, str(script_path)] + list(script_args)
        
        print(f"WORKER: Executing script: {' '.join(cmd)}", flush=True)
        if working_dir:
            print(f"WORKER: Working directory: {working_dir}", flush=True)
        
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_dir,
            env=env,
            text=True,
            bufsize=1,  # Line buffered
        )
        
        stdout_lines = []
        stderr_lines = []
        last_progress = 0.0
        training_complete = False
        
        # Monitor stdout for progress updates
        # Format expected: "PROGRESS:XX:message" where XX is 0-100
        progress_pattern = re.compile(r"PROGRESS:(\d+(?:\.\d+)?):?(.*)")
        
        def read_output():
            nonlocal last_progress, training_complete
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                    
                line = line.rstrip()
                stdout_lines.append(line)
                print(f"SCRIPT: {line}", flush=True)
                
                # Parse progress updates
                match = progress_pattern.match(line)
                if match:
                    progress = float(match.group(1))
                    message = match.group(2).strip() if match.group(2) else ""
                    last_progress = progress
                    STATE.send_progress(progress, message)
                    
                if "TRAINING_COMPLETE" in line:
                    training_complete = True
                    
        def read_stderr():
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                line = line.rstrip()
                stderr_lines.append(line)
                # Only print warnings/errors to not spam logs
                if "error" in line.lower() or "warning" in line.lower() or "traceback" in line.lower():
                    print(f"SCRIPT ERROR: {line}", flush=True)
        
        # Start reader threads
        stdout_thread = threading.Thread(target=read_output, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process with timeout
        try:
            return_code = process.wait(timeout=timeout_secs)
        except subprocess.TimeoutExpired:
            process.kill()
            return {
                "status": "error",
                "error": f"Script timed out after {timeout_secs} seconds",
                "stdout": "\n".join(stdout_lines[-100:]),  # Last 100 lines
                "stderr": "\n".join(stderr_lines[-100:]),
            }
        
        # Wait for reader threads
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        
        # Check result
        if return_code != 0:
            return {
                "status": "error",
                "error": f"Script exited with code {return_code}",
                "return_code": return_code,
                "stdout": "\n".join(stdout_lines),
                "stderr": "\n".join(stderr_lines),
            }
        
        STATE.send_progress(100.0, "Script completed successfully")
        
        return {
            "status": "ok",
            "return_code": 0,
            "training_complete": training_complete,
            "final_progress": last_progress,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines) if stderr_lines else None,
        }
        
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def handle_request(request: Dict[str, Any], client_id: Optional[str] = None) -> Dict[str, Any]:
    """Handle incoming request"""
    cmd = request.get("cmd", "")
    
    if cmd == "ping":
        queue_status = JOB_QUEUE.get_queue_status()
        return {
            "status": "ok",
            "message": "pong",
            "device": STATE.device,
            "model_loaded": STATE.current_model_name,
            "cuda_available": torch.cuda.is_available(),
            "queue": queue_status,
        }
    
    # ========================================================================
    # Job Queue Commands
    # ========================================================================
    
    elif cmd == "submit_job":
        # Queue a training job (non-blocking)
        job_cmd = request.get("job_cmd", "train")  # "train" or "run_script"
        data = request.get("data", {})
        job_id = JOB_QUEUE.submit(job_cmd, data, client_id)
        return {
            "status": "ok",
            "job_id": job_id,
            "message": f"Job {job_id} queued",
            "queue": JOB_QUEUE.get_queue_status(),
        }
    
    elif cmd == "job_status":
        # Get status of a specific job
        job_id = request.get("job_id")
        if not job_id:
            return {"status": "error", "error": "job_id required"}
        job = JOB_QUEUE.get_job(job_id)
        if not job:
            return {"status": "error", "error": f"Job {job_id} not found"}
        return {
            "status": "ok",
            "job": job.to_dict(),
        }
    
    elif cmd == "list_jobs":
        # List all jobs
        limit = request.get("limit", 50)
        jobs = JOB_QUEUE.list_jobs(limit)
        return {
            "status": "ok",
            "jobs": jobs,
            "queue": JOB_QUEUE.get_queue_status(),
        }
    
    elif cmd == "cancel_job":
        # Cancel a pending job
        job_id = request.get("job_id")
        if not job_id:
            return {"status": "error", "error": "job_id required"}
        if JOB_QUEUE.cancel_job(job_id):
            return {"status": "ok", "message": f"Job {job_id} cancelled"}
        else:
            return {"status": "error", "error": f"Cannot cancel job {job_id} (may be running or completed)"}
    
    elif cmd == "queue_status":
        # Get queue statistics
        return {
            "status": "ok",
            "queue": JOB_QUEUE.get_queue_status(),
        }
    
    # ========================================================================
    # Synchronous Commands (backward compatibility - blocks until complete)
    # ========================================================================
        
    elif cmd == "train":
        # Synchronous training (blocks until complete)
        return process_training_request(request.get("data", {}))
        
    elif cmd == "run_script":
        # Synchronous script execution (blocks until complete)
        return run_local_script(request.get("data", {}))
        
    elif cmd == "unload":
        # Unload model to free memory
        if STATE.model is not None:
            del STATE.model
            STATE.model = None
            STATE.current_model_name = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return {"status": "ok", "message": "Model unloaded"}
        
    elif cmd == "shutdown":
        STATE.running = False
        return {"status": "ok", "message": "Shutting down"}
        
    else:
        return {"status": "error", "error": f"Unknown command: {cmd}"}


def handle_client(conn: socket.socket, addr, client_id: str):
    """Handle a single client connection in its own thread"""
    print(f"WORKER: Client {client_id} connected from {addr}", flush=True)
    STATE.register_client(client_id, conn)
    
    buffer = b""
    try:
        while STATE.running:
            try:
                conn.settimeout(1.0)
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data
                
                # Check for complete messages
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line:
                        try:
                            request = json.loads(line.decode())
                            response = handle_request(request, client_id)
                            response["type"] = "result"
                            conn.sendall((json.dumps(response) + "\n").encode())
                        except json.JSONDecodeError as e:
                            error_response = {"type": "result", "status": "error", "error": f"Invalid JSON: {e}"}
                            conn.sendall((json.dumps(error_response) + "\n").encode())
                            
            except socket.timeout:
                continue
            except Exception as e:
                print(f"WORKER: Error handling client {client_id}: {e}", flush=True)
                break
    finally:
        STATE.unregister_client(client_id)
        try:
            conn.close()
        except:
            pass
        print(f"WORKER: Client {client_id} disconnected", flush=True)


def run_server(address: str):
    """Run the worker server
    
    Args:
        address: Either a Unix socket path (e.g., /tmp/worker.sock) 
                 or TCP address (e.g., 0.0.0.0:9500 or tcp:0.0.0.0:9500)
    """
    STATE.socket_path = address
    
    # Determine socket type
    use_tcp = address.startswith("tcp:") or ":" in address and not address.startswith("/")
    
    if use_tcp:
        # TCP socket
        addr = address.replace("tcp:", "")
        if ":" in addr:
            host, port = addr.rsplit(":", 1)
            port = int(port)
        else:
            host = "0.0.0.0"
            port = int(addr)
        
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(10)  # Allow more pending connections
        server.settimeout(1.0)
        
        print(f"WORKER: Training worker listening on TCP {host}:{port}", flush=True)
    else:
        # Unix socket
        if os.path.exists(address):
            os.unlink(address)
            
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(address)
        server.listen(10)  # Allow more pending connections
        server.settimeout(1.0)
        
        print(f"WORKER: Training worker listening on Unix socket {address}", flush=True)
    
    print(f"WORKER: Device: {STATE.device}, CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"WORKER: GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"WORKER: GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)
    
    # Start job processor thread
    processor_thread = threading.Thread(target=job_processor, daemon=True)
    processor_thread.start()
    print("WORKER: Job queue enabled - multiple clients can submit jobs", flush=True)
    
    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("WORKER: Received shutdown signal", flush=True)
        STATE.running = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    client_threads: List[threading.Thread] = []
    client_counter = 0
    
    while STATE.running:
        try:
            conn, addr = server.accept()
            client_counter += 1
            client_id = f"client_{client_counter}"
            
            # Handle client in separate thread
            thread = threading.Thread(
                target=handle_client,
                args=(conn, addr, client_id),
                daemon=True
            )
            thread.start()
            client_threads.append(thread)
            
            # Cleanup finished threads periodically
            client_threads = [t for t in client_threads if t.is_alive()]
            
        except socket.timeout:
            continue
        except Exception as e:
            if STATE.running:
                print(f"WORKER: Server error: {e}", flush=True)
    
    # Wait for threads to finish
    print("WORKER: Waiting for client threads to finish...", flush=True)
    for thread in client_threads:
        thread.join(timeout=5.0)
                
    # Cleanup
    server.close()
    # Only unlink if it's a Unix socket path (not TCP)
    if not use_tcp and os.path.exists(address):
        os.unlink(address)
    print("WORKER: Shutdown complete", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: training_worker.py <address>")
        print("  Unix socket: training_worker.py /tmp/worker.sock")
        print("  TCP socket:  training_worker.py 0.0.0.0:9500")
        print("  TCP socket:  training_worker.py tcp:0.0.0.0:9500")
        sys.exit(1)
        
    address = sys.argv[1]
    run_server(address)
