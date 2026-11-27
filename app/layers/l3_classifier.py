"""L3 semantic classifier: DeBERTa prompt-injection model on local ONNX runtime (IP-1, IP-2, IP-3, IP-4, IP-5, IP-7, IP-9, TM-R2).

Model: any DeBERTa sequence-classification ONNX export with a `tokenizer.json`
and a `config.json` naming the malicious label (`L3_MODEL_DIR`). The default
is the off-the-shelf `protectai/deberta-v3-base-prompt-injection-v2` export,
which runs on CPU with no training; a Colab fine-tuned export (§2.7) replaces
it by pointing `L3_MODEL_DIR` at the new directory, and its hashes change the
layer version (IP-5).

Scoring (IP-1, IP-2): L3 reads the `deberta` window plan of the shared
InspectionBundle, where every window of both views was already split to fit
DeBERTa's 512-token limit (`truncation_count` stays 0). Each window gets the
softmax probability of the malicious label; a view's score is the maximum over
its windows, and the view blocks when it reaches `threshold` (default 0.5).
A block on either view blocks the request. Identical window texts are scored
once per call.

Scan mode (IP-3, IP-4, Step 5B.1 M-2):

- `early_stop=True` (production and `eval`): the scan stops at the first
  window that reaches the threshold and returns `block` at once, so a block is
  never lost to a later window hitting the deadline. The result then holds
  every view that was scanned completely plus the blocking view (with the
  scores of the windows scored before the stop); a view left incomplete
  without a block is not reported.
- `early_stop=False` (both shadow tiers): every window of both views is
  scored for full telemetry.

The scan mode is part of the layer version, so the IP-5 layer-verdict cache
never replays an early-stopped result into a shadow run.

Budget (TM-R2, IP-7): inference runs on the bounded layer pool; with a
`budget_ms` the layer checks its deadline after every window (after the block
check) and stops with `LayerBudgetExceeded` (a `timeout`) once the L3 budget
(default 150 ms) is spent, which frees the pool thread instead of finishing
the remaining windows. The shadow tiers build L3 with `budget_ms=None`, so no
deadline is enforced there (IP-4, Step 5B.1 H2).

TM-R3 tightens only the L2 threshold, so `degraded` does not change L3.
"""

from __future__ import annotations

from app.layers.base import Deadline, view_results
from app.onnx_models import Classifier, tokenizer_limit_for
from contracts.hashing import canonical_hash
from contracts.inspection import InspectionBundle, TextWindow
from contracts.layer_result import LayerResult

L3_REVISION = 2
DEFAULT_L3_THRESHOLD = 0.5


class L3ClassifierLayer:
    """DeBERTa malicious-probability layer (L3)."""

    name = "L3"
    enabled = True

    def __init__(
        self,
        classifier: Classifier,
        *,
        threshold: float = DEFAULT_L3_THRESHOLD,
        budget_ms: float | None = None,
        early_stop: bool = True,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("L3 threshold must be a probability in (0, 1]")
        self.classifier = classifier
        self.threshold = threshold
        self.budget_ms = budget_ms
        self.early_stop = early_stop
        self.tokenizer_limit = tokenizer_limit_for(classifier)
        self.version = canonical_hash(
            {
                "layer": "L3",
                "revision": L3_REVISION,
                "classifier_version": classifier.version,
                "tokenizer": classifier.name,
                "token_limit": classifier.token_limit,
                "threshold": threshold,
                "score": "max-malicious-probability",
                "scan": "early-stop" if early_stop else "all-windows",
            }
        )

    @property
    def chroma_version(self) -> str | None:
        return None

    def inspect(self, bundle: InspectionBundle, *, degraded: bool = False) -> tuple[LayerResult, ...]:
        """One result per reported view (raw, then decoded). `degraded` is ignored: TM-R3 tightens L2 only."""
        deadline = Deadline(self.budget_ms)
        plan = bundle.plan_for(self.classifier.name)
        views: tuple[tuple[str, tuple[TextWindow, ...]], ...] = (
            ("raw_view", plan.raw_windows),
            ("decoded_view", plan.decoded_windows),
        )
        scores: dict[str, float] = {}
        stopped = False
        for _, windows in views:
            for window in windows:
                if window.text in scores:
                    continue
                scores[window.text] = self.classifier.malicious_probability(window.text)
                if self.early_stop and scores[window.text] >= self.threshold:
                    stopped = True  # IP-3: the block is decided; later windows are not needed
                    break
                deadline.check()
            if stopped:
                break
        return view_results(self.name, views, scores, self.threshold, deadline.elapsed_ms())
