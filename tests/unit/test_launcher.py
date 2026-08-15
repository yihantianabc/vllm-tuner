"""Unit tests for the vLLM launcher."""

from unittest.mock import Mock

import httpx
import pytest

from vllm_tuner.config.models import TuningConfig
from vllm_tuner.vllm.launcher import VLLMLauncher


@pytest.mark.asyncio
async def test_wait_ready_retries_transient_request_error(monkeypatch):
    """Retry transient protocol errors while the server is starting."""
    launcher = VLLMLauncher(TuningConfig(model="test-model"))
    launcher.process = Mock()
    launcher.process.poll.return_value = None

    class FakeResponse:
        status_code = 200

    class FakeAsyncClient:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def get(self, url):
            self.__class__.calls += 1
            if self.__class__.calls == 1:
                raise httpx.RemoteProtocolError("server is still starting")
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    assert await launcher.wait_ready(timeout=1, check_interval=0.01)
    assert FakeAsyncClient.calls == 2
