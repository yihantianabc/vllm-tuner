"""Benchmark workload definitions."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..config.models import WorkloadConfig


class Workload(ABC):
    """Base class for benchmark workloads."""

    def __init__(self, config: WorkloadConfig):
        self.config = config
        self._prompts: Optional[List[str]] = None

    @abstractmethod
    async def load(self) -> List[str]:
        """Load prompts for the workload."""
        pass

    async def get_prompts(self) -> List[str]:
        """Get prompts, loading if necessary."""
        if self._prompts is None:
            self._prompts = await self.load()
        return self._prompts

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get workload metadata."""
        pass

    def unload(self) -> None:
        """Unload prompts to free memory."""
        self._prompts = None
