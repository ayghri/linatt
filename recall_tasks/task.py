"""In-context recall tasks (Arora'24 / Based cloze-completion) for TriviaQA and NQ,
matching the GDN paper's recall suite. lm-eval ships based-squad/swde/fda but NOT these two,
so eval_recall.py fell back to the CLOSED-BOOK triviaqa/nq_open (rc.nocontext) -> ~0 at 340M.

Each doc = [passage/context] + [question rewritten as a declarative cloze prefix] -> answer span
that appears in the context. Scored by `contains` (normalized substring match), generate_until "\n",
max_gen_toks 48 -- identical mechanics to lm_eval.tasks.squad_completion.
Datasets: hazyresearch/based_triviaqa, hazyresearch/based_nq_512.
"""
import re
from copy import deepcopy
from typing import List

import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask


def contains_score(prediction: str, labels: List[str]):
    return max(
        int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), prediction)))
        for label in labels
    )


class _BasedRecallQA(ConfigurableTask):
    """In-context cloze recall. Subclasses set DATASET_PATH and PAR."""
    VERSION = 0
    DATASET_NAME = "default"
    PAR = False   # True -> convert "[PAR]" paragraph markers to newlines (TriviaQA)

    def __init__(self, **kwargs):
        super().__init__(config={"metadata": {"version": self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        ctx = doc["context"]
        if self.PAR:
            ctx = ctx.replace(" [PAR] ", "\n\n").replace("[PAR]", "\n\n")
        return ctx.strip() + "\n" + doc["question"]        # question is already a cloze prefix

    def doc_to_target(self, doc):
        return doc["answers"][0]

    def construct_requests(
        self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs
    ):
        arguments = deepcopy(self.config.generation_kwargs) if self.config.generation_kwargs else {}
        arguments["until"] = arguments.get("until", ["\n"])
        arguments["max_gen_toks"] = arguments.get("max_gen_toks", 48)
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, arguments),
                idx=0,
                **kwargs,
            )
        ]

    def process_results(self, doc, results):
        return {"contains": contains_score(results[0], doc["answers"])}   # any gold answer in gen

    def aggregation(self):
        return {"contains": np.mean}

    def higher_is_better(self):
        return {"contains": True}


class BasedTriviaQA(_BasedRecallQA):
    DATASET_PATH = "hazyresearch/based_triviaqa"
    PAR = True


class BasedNQ(_BasedRecallQA):
    DATASET_PATH = "hazyresearch/based_nq_512"
    PAR = False
