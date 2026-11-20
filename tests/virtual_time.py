"""A virtual wall clock for clock tests.

`sleep` parks the caller until virtual time reaches its wake-up; once every task is parked, time jumps to the earliest
wake-up. Several tasks (the clock, a fake brain, a slow fake engine) can share it, schedules are exact, and tests take
milliseconds.
"""

import asyncio
import heapq
import itertools


class VirtualTime:
    def __init__(self) -> None:
        self.t = 0.0
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()
        self._advancer: asyncio.Task[None] | None = None

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        future = asyncio.get_running_loop().create_future()
        heapq.heappush(self._sleepers, (self.t + max(seconds, 0.0), next(self._seq), future))
        if self._advancer is None or self._advancer.done():
            self._advancer = asyncio.create_task(self._advance())
        await future

    async def _advance(self) -> None:
        while self._sleepers:
            for _ in range(20):  # let every runnable task reach its next sleep
                await asyncio.sleep(0)
            wake, _, future = heapq.heappop(self._sleepers)
            self.t = max(self.t, wake)
            if not future.done():
                future.set_result(None)


class VirtualLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock is virtual: when nothing is runnable, time jumps to the next timer.

    `loop.time()`, `asyncio.sleep` and `asyncio.wait(timeout=...)` all run on it, so code that reads the running loop's
    time (the lockstep clock's hold and brain-timeout bookkeeping) is exact and instant under test.
    """

    def __init__(self) -> None:
        super().__init__()
        self._virtual_t = 0.0

    def time(self) -> float:
        return self._virtual_t

    def _run_once(self) -> None:
        if not self._ready and self._scheduled:
            self._virtual_t = max(self._virtual_t, self._scheduled[0]._when)
        super()._run_once()


def run_virtual(make_coro):
    """Run `make_coro(loop)` to completion on a fresh `VirtualLoop` and return its result."""
    loop = VirtualLoop()
    try:
        return loop.run_until_complete(make_coro(loop))
    finally:
        loop.close()
