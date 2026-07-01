from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentdojo import agent_pipeline, functions_runtime


class MyAgentPreProcessor(agent_pipeline.BasePipelineElement):
    name = "my_agent_preprocessor"

    def __init__(self, task_shield: Any | None = None):
        self.task_shield = task_shield

    def query(self, query, runtime, env=functions_runtime.EmptyEnv(), messages=None, extra_args=None):
        if messages is None:
            messages = []
        if extra_args is None:
            extra_args = {}

        extra_args = dict(extra_args)
        extra_args["original_user_query"] = query
        if self.task_shield is not None:
            self.task_shield.ensure_user_task_set(query, runtime, env, messages, extra_args)
        return query, runtime, env, messages, extra_args


class DefenseHook(ABC):

    @abstractmethod
    def query(self, query, runtime, env=functions_runtime.EmptyEnv(), messages=None, extra_args=None):
        raise NotImplementedError
