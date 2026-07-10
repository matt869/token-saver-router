"""Local model wrapper (free / zero-scored path).

Runs a HuggingFace ``transformers`` causal LM on the local GPU. Under ROCm, AMD
GPUs are exposed to PyTorch as the ``cuda`` device, so the default device string
is ``"cuda"`` (configurable via ``LOCAL_DEVICE``).

To keep the memory footprint small (and avoid host-RAM / paging blow-ups when a
7B model is materialised), the model can be loaded in **4-bit** via
``bitsandbytes`` (``BitsAndBytesConfig``). This drops Qwen2.5-7B from ~15-16 GB
(bf16) to ~5-6 GB of VRAM — the difference between fitting and not fitting on a
16 GB AMD card.

**Environment-aware defaults.** The 4-bit path is meant for the Linux/ROCm
production box, not the Windows dev laptop (where the stock ``bitsandbytes``
wheel is CUDA-only and can crash on import). So, when nothing is set explicitly,
4-bit turns on automatically only in *production* and stays off everywhere else:

* Explicit ``load_in_4bit=...`` constructor arg  → always wins.
* Explicit ``LOAD_IN_4BIT`` env var              → wins over auto-detection.
* Otherwise: 4-bit ON when the environment is *production*
  (``ENV``/``APP_ENV``/``ENVIRONMENT`` = ``production``), OFF otherwise.

Even when 4-bit is requested, the ``bitsandbytes`` import is guarded: if it (or
its GPU kernels) is missing, we log a clear install hint and fall back to
bf16/fp32 rather than crashing. So the same code runs smoothly on the Windows
dev box and locks into 4-bit ROCm in production.

The heavy imports (``torch``, ``transformers``, ``bitsandbytes``) happen lazily
inside :meth:`LocalModel.load` so this module — and therefore the classifier and
its tests — can be imported on a machine with no GPU and no ML stack installed.
"""

from __future__ import annotations

import os
import platform
import threading
from dataclasses import dataclass
from typing import Optional

_PROD_ALIASES = {"prod", "production", "live", "staging"}
_DEV_ALIASES = {"dev", "development", "local", "test", "testing"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def resolve_environment() -> str:
    """Return ``"production"`` or ``"development"`` from the environment.

    Reads ``ENV`` first, then ``APP_ENV`` / ``ENVIRONMENT`` as aliases. Anything
    unrecognised (or unset) is treated as development — the safe default for the
    Windows dev machine.
    """

    for var in ("ENV", "APP_ENV", "ENVIRONMENT"):
        val = os.getenv(var)
        if val:
            v = val.strip().lower()
            if v in _PROD_ALIASES:
                return "production"
            if v in _DEV_ALIASES:
                return "development"
    return "development"


def _default_load_in_4bit() -> bool:
    """Decide the 4-bit default when the caller doesn't specify one.

    Precedence: explicit ``LOAD_IN_4BIT`` env var wins; otherwise enable 4-bit
    only in production. This keeps local (Windows) testing on the safe bf16/fp32
    path while the Linux/ROCm deployment locks into 4-bit automatically.
    """

    raw = os.getenv("LOAD_IN_4BIT")
    if raw not in (None, ""):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return resolve_environment() == "production"


@dataclass
class LocalResult:
    """Answer plus the number of tokens the local model produced.

    These tokens are *free*: they count as zero toward the hackathon score. We
    still track them so the eval harness can report the local/remote split.
    """

    text: str
    tokens: int  # prompt + completion tokens on the local path (scored as 0)


class LocalModel:
    """Lazily-loaded local causal LM."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 256,
        load_in_4bit: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.device = device
        # Preserved verbatim: load() may mutate self.device to "cpu" on GPU
        # fallback, but /health still needs to show what was *asked* for.
        self._configured_device = device
        self.max_new_tokens = max_new_tokens
        self.environment = resolve_environment()
        # 4-bit default is environment-aware: explicit arg wins, else the
        # LOAD_IN_4BIT env var, else ON only in production. Only actually applied
        # on a GPU device, and only if bitsandbytes imports cleanly — see load().
        if load_in_4bit is None:
            load_in_4bit = _default_load_in_4bit()
        self.load_in_4bit = load_in_4bit
        self._quantized = False  # set True once weights are loaded in 4-bit
        # Resolved at load() time so /health can report what the box is REALLY
        # running on (e.g. cpu/fp32 after a silent GPU fallback), not just what
        # LOCAL_DEVICE was configured to.
        self._resolved_device: Optional[str] = None
        self._resolved_dtype: Optional[str] = None
        self._tokenizer = None
        self._model = None
        self._torch = None
        # FastAPI serves sync endpoints from a threadpool; serialise load and
        # generation (concurrent GPU generate risks OOM and garbled batching).
        # RLock because generate() calls load().
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def runtime_info(self) -> dict:
        """Resolved device/dtype for the /health endpoint.

        Once the weights are loaded this reports what the model is *really*
        running on — e.g. ``cpu``/``fp32`` after a silent GPU fallback — so a
        glance at /health tells you whether the demo box is on the GPU or not.
        Before the first generation it reports the *configured* device and a
        note that resolution happens lazily (the device may still fall back).
        """

        if self.loaded:
            return {
                "loaded": True,
                "configured_device": self._configured_device,
                "device": self._resolved_device,
                "dtype": self._resolved_dtype,
                "quantized": self._quantized,
            }
        return {
            "loaded": False,
            "configured_device": self._configured_device,
            "device": None,
            "dtype": "unloaded (resolves at first generation)",
            "quantized": False,
        }

    def load(self) -> None:
        """Import the ML stack and materialise the model/tokenizer on device.

        Idempotent: the ``self.loaded`` guard means the weights are materialised
        exactly once per process. The LLM judge (``LLMClassifier``) and the
        answering path share this same instance, so judging + answering never
        load the model twice.
        """

        with self._lock:
            if self.loaded:
                return

            import torch  # noqa: WPS433 (lazy import is intentional)
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            on_gpu = str(self.device).startswith("cuda")
            # Graceful GPU degradation: if a cuda/ROCm device was requested but
            # torch can't see one, fall back to CPU instead of raising a
            # RuntimeError on .to(device). ROCm presents AMD GPUs as "cuda" and
            # reports is_available()==True, so the GPU path is preserved on the
            # ROCm box while a CPU-only dev machine is rescued.
            if on_gpu and not torch.cuda.is_available():
                print(
                    f"[local_model] device '{self.device}' requested but no GPU "
                    "is visible to torch -> falling back to CPU (fp32)."
                )
                self.device = "cpu"
                on_gpu = False

            want_4bit = self.load_in_4bit and on_gpu
            if self.load_in_4bit and not on_gpu:
                print(
                    f"[local_model] 4-bit requested but device is '{self.device}'"
                    " (not a GPU) -> loading in fp32."
                )

            # Try to build the 4-bit config; a missing/broken bitsandbytes must
            # never crash the process (crucial on the Windows dev box, where the
            # stock CUDA wheel can fail to import). On failure we warn + fall back.
            quant_config = self._maybe_build_4bit_config(torch) if want_4bit else None

            if quant_config is not None:
                # device_map lets accelerate + bitsandbytes place the quantized
                # weights directly on the GPU. IMPORTANT: a 4-bit model is
                # already placed and MUST NOT be moved with .to() afterwards
                # (transformers raises if you try), so there is no .to() here.
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    device_map={"": self.device},
                )
                self._quantized = True
                self._resolved_device = str(self.device)
                self._resolved_dtype = "4bit-nf4"
                print(
                    f"[local_model] loaded '{self.model_name}' in 4-bit NF4 on "
                    f"{self.device} (env={self.environment})"
                )
            else:
                # Fallback: bf16 on GPU (half the VRAM of fp32, better range than
                # fp16), fp32 on CPU (bf16 CPU inference is slow). Used when 4-bit
                # is off (dev default), on CPU, or when bitsandbytes is missing.
                dtype = torch.bfloat16 if on_gpu else torch.float32
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                ).to(self.device)
                self._quantized = False
                self._resolved_device = str(self.device)
                self._resolved_dtype = "bf16" if on_gpu else "fp32"
                print(
                    f"[local_model] loaded '{self.model_name}' in "
                    f"{'bf16' if on_gpu else 'fp32'} on {self.device} "
                    f"(env={self.environment}, 4-bit={'off' if not want_4bit else 'unavailable'})"
                )

            self._model.eval()

    def _maybe_build_4bit_config(self, torch):
        """Build a 4-bit ``BitsAndBytesConfig``, or ``None`` if unavailable.

        Guards the ``bitsandbytes`` import so a missing library or missing GPU
        kernels degrade to bf16/fp32 instead of crashing. On the Windows dev box
        this is the difference between "runs" and "OSError on import"; in
        production the import succeeds and 4-bit engages.
        """

        try:
            import bitsandbytes  # noqa: F401,WPS433 — probe that kernels import
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # noqa: BLE001 — any import failure -> fall back
            hint = (
                "pip install bitsandbytes accelerate"
                if platform.system() != "Linux"
                else "install the ROCm build of bitsandbytes (multi-backend): "
                "https://huggingface.co/docs/bitsandbytes/main/en/installation#amd-gpu"
            )
            print(
                f"[local_model] 4-bit requested but bitsandbytes is unavailable "
                f"({exc.__class__.__name__}: {exc}). Falling back to bf16/fp32. "
                f"To enable 4-bit: {hint}"
            )
            return None

        # 4-bit (NF4): ~1/4 the VRAM of bf16, so Qwen2.5-7B fits in ~5-6 GB and
        # stops thrashing host RAM / the paging file at load. Compute stays in
        # bf16 for accuracy; double-quant also quantizes the quant constants.
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> LocalResult:
        """Generate an answer and report tokens used (prompt + completion)."""

        self._lock.acquire()
        try:
            return self._generate_locked(prompt, max_new_tokens)
        finally:
            self._lock.release()

    def generate_samples(self, prompt: str, n: int, max_new_tokens: Optional[int] = None):
        """Draw ``n`` *stochastic* samples for self-consistency checking.

        Greedy decoding would return ``n`` identical answers, which can't reveal
        uncertainty — so these use temperature sampling. Returns a list of
        :class:`LocalResult`. The lock is held across all samples so concurrent
        requests don't interleave on the GPU.
        """

        n = max(1, int(n))
        self._lock.acquire()
        try:
            return [self._generate_locked(prompt, max_new_tokens, sample=True) for _ in range(n)]
        finally:
            self._lock.release()

    def _generate_locked(
        self, prompt: str, max_new_tokens: Optional[int] = None, sample: bool = False,
    ) -> LocalResult:
        self.load()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        budget = max_new_tokens or self.max_new_tokens

        # Use the chat template when the tokenizer provides one (Gemma does).
        if getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            encoded = tokenizer(prompt, return_tensors="pt")

        # transformers >=5 returns a dict-like BatchEncoding here; older versions
        # returned a bare tensor. Normalise to (input_ids, attention_mask).
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded.to(self.device)
            attention_mask = None
        else:
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

        prompt_len = int(input_ids.shape[-1])

        gen_kwargs = {"max_new_tokens": budget, "do_sample": sample}
        if sample:
            # Temperature sampling gives the diversity self-consistency needs.
            gen_kwargs.update({"temperature": 0.7, "top_p": 0.95})
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            output_ids = model.generate(input_ids, **gen_kwargs)

        completion_ids = output_ids[0][prompt_len:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        total_tokens = int(output_ids.shape[-1])  # prompt + completion
        return LocalResult(text=text, tokens=total_tokens)
