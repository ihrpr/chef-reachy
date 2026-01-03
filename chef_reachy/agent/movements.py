"""Shared movement utilities for Reachy Mini robot expressions."""

import asyncio
import logging
import random
import threading
import time
from typing import Any

import numpy as np
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import InterpolationTechnique

logger = logging.getLogger(__name__)


class RobotMovements:
    """Manages expressive movements for Reachy Mini robot."""

    def __init__(self, reachy_mini: Any):
        """Initialize robot movements.

        Args:
            reachy_mini: ReachyMini robot instance
        """
        self.reachy_mini = reachy_mini

    async def nod(self) -> None:
        """Execute a nod movement (agreement, acknowledgment)."""
        try:
            head_down = create_head_pose(z=-20, mm=True)
            head_up = create_head_pose(z=15, mm=True)
            neutral = create_head_pose(z=0, mm=True)

            self.reachy_mini.goto_target(
                head=head_down,
                antennas=np.deg2rad([25, 25]),
                duration=0.5,
                method=InterpolationTechnique.MIN_JERK,
            )
            await asyncio.sleep(0.5)
            self.reachy_mini.goto_target(
                head=head_up,
                antennas=np.deg2rad([35, 35]),
                duration=0.5,
                method=InterpolationTechnique.MIN_JERK,
            )
            await asyncio.sleep(0.5)
            self.reachy_mini.goto_target(
                head=neutral,
                antennas=np.deg2rad([0, 0]),
                duration=0.4,
                method=InterpolationTechnique.MIN_JERK,
            )
        except Exception as e:
            logger.error(f"Error executing nod: {e}")

    async def no_shake(self) -> None:
        """Execute a head shake (disagreement, concern)."""
        try:
            neutral = create_head_pose(z=0, mm=True)

            self.reachy_mini.goto_target(
                head=neutral,
                body_yaw=np.deg2rad(30),
                antennas=np.deg2rad([20, 20]),
                duration=0.4,
                method=InterpolationTechnique.MIN_JERK,
            )
            await asyncio.sleep(0.4)
            self.reachy_mini.goto_target(
                head=neutral,
                body_yaw=np.deg2rad(-30),
                antennas=np.deg2rad([20, 20]),
                duration=0.5,
                method=InterpolationTechnique.MIN_JERK,
            )
            await asyncio.sleep(0.5)
            self.reachy_mini.goto_target(
                head=neutral,
                body_yaw=0.0,
                antennas=np.deg2rad([0, 0]),
                duration=0.4,
                method=InterpolationTechnique.MIN_JERK,
            )
        except Exception as e:
            logger.error(f"Error executing no_shake: {e}")

    async def greet(self) -> None:
        """Execute a greeting gesture."""
        try:
            neutral = create_head_pose(z=0, mm=True)

            self.reachy_mini.goto_target(
                head=neutral,
                antennas=np.deg2rad([70, 70]),
                duration=0.6,
                method=InterpolationTechnique.MIN_JERK,
            )
            await asyncio.sleep(0.6)
            self.reachy_mini.goto_target(
                antennas=np.deg2rad([35, 35]),
                duration=0.5,
                method=InterpolationTechnique.MIN_JERK,
            )
            await asyncio.sleep(0.5)
            self.reachy_mini.goto_target(
                antennas=np.deg2rad([0, 0]),
                duration=0.4,
                method=InterpolationTechnique.MIN_JERK,
            )
        except Exception as e:
            logger.error(f"Error executing greet: {e}")

    async def thinking(self) -> None:
        """Execute thinking/processing movement (alternating antennas)."""
        try:
            self.reachy_mini.goto_target(
                antennas=np.deg2rad([55, -55]),
                duration=0.5,
                method=InterpolationTechnique.MIN_JERK,
            )
            await asyncio.sleep(0.5)
            self.reachy_mini.goto_target(
                antennas=np.deg2rad([-55, 55]),
                duration=0.6,
                method=InterpolationTechnique.MIN_JERK,
            )
            await asyncio.sleep(0.6)
            self.reachy_mini.goto_target(
                antennas=np.deg2rad([0, 0]),
                duration=0.4,
                method=InterpolationTechnique.MIN_JERK,
            )
        except Exception as e:
            logger.error(f"Error executing thinking: {e}")

    def trigger_movement_async(self, movement_type: str) -> None:
        """Trigger a movement in fire-and-forget mode (async context).

        Args:
            movement_type: One of "nod", "no_shake", "greet", "thinking"
        """

        async def execute() -> None:
            if movement_type == "nod":
                await self.nod()
            elif movement_type == "no_shake":
                await self.no_shake()
            elif movement_type == "greet":
                await self.greet()
            elif movement_type == "thinking":
                await self.thinking()

        asyncio.create_task(execute())

    def trigger_movement_threaded(self, movement_type: str) -> None:
        """Trigger a movement in fire-and-forget mode (threaded context).

        Use this when called from a thread without an asyncio event loop (e.g., TTS worker).

        Args:
            movement_type: One of "nod", "no_shake", "greet", "thinking", or speaking movements
        """

        def execute() -> None:
            if movement_type == "nod":
                self._nod_sync()
            elif movement_type == "no_shake":
                self._no_shake_sync()
            elif movement_type == "greet":
                self._greet_sync()
            elif movement_type == "thinking":
                self._thinking_sync()
            # Speaking-specific movements
            elif movement_type == "head_tilt":
                self._head_tilt_sync()
            elif movement_type == "antenna_wiggle":
                self._antenna_wiggle_sync()
            elif movement_type == "antenna_wave":
                self._antenna_wave_sync()
            elif movement_type == "head_bob":
                self._head_bob_sync()

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()

    def random_speaking_movement(self) -> None:
        """Trigger a random speaking movement (for TTS).

        Called from TTS worker thread, so uses threaded execution.
        """
        movement = random.choice([
            "head_tilt",
            "antenna_wiggle",
            "nod",
            "antenna_wave",
            "head_bob",
        ])
        self.trigger_movement_threaded(movement)

    # Synchronous versions for threaded execution

    def _nod_sync(self) -> None:
        """Synchronous nod movement."""
        try:
            head_down = create_head_pose(z=-20, mm=True)
            head_up = create_head_pose(z=15, mm=True)
            neutral = create_head_pose(z=0, mm=True)

            self.reachy_mini.goto_target(head=head_down, antennas=np.deg2rad([25, 25]), duration=0.5, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.5)
            self.reachy_mini.goto_target(head=head_up, antennas=np.deg2rad([35, 35]), duration=0.5, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.5)
            self.reachy_mini.goto_target(head=neutral, antennas=np.deg2rad([0, 0]), duration=0.4, method=InterpolationTechnique.MIN_JERK)
        except Exception as e:
            logger.debug(f"Error executing nod: {e}")

    def _no_shake_sync(self) -> None:
        """Synchronous no shake movement."""
        try:
            neutral = create_head_pose(z=0, mm=True)

            self.reachy_mini.goto_target(head=neutral, body_yaw=np.deg2rad(30), antennas=np.deg2rad([20, 20]), duration=0.4, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.4)
            self.reachy_mini.goto_target(head=neutral, body_yaw=np.deg2rad(-30), antennas=np.deg2rad([20, 20]), duration=0.5, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.5)
            self.reachy_mini.goto_target(head=neutral, body_yaw=0.0, antennas=np.deg2rad([0, 0]), duration=0.4, method=InterpolationTechnique.MIN_JERK)
        except Exception as e:
            logger.debug(f"Error executing no_shake: {e}")

    def _greet_sync(self) -> None:
        """Synchronous greet movement."""
        try:
            neutral = create_head_pose(z=0, mm=True)

            self.reachy_mini.goto_target(head=neutral, antennas=np.deg2rad([70, 70]), duration=0.6, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.6)
            self.reachy_mini.goto_target(antennas=np.deg2rad([35, 35]), duration=0.5, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.5)
            self.reachy_mini.goto_target(antennas=np.deg2rad([0, 0]), duration=0.4, method=InterpolationTechnique.MIN_JERK)
        except Exception as e:
            logger.debug(f"Error executing greet: {e}")

    def _thinking_sync(self) -> None:
        """Synchronous thinking movement."""
        try:
            self.reachy_mini.goto_target(antennas=np.deg2rad([55, -55]), duration=0.5, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.5)
            self.reachy_mini.goto_target(antennas=np.deg2rad([-55, 55]), duration=0.6, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.6)
            self.reachy_mini.goto_target(antennas=np.deg2rad([0, 0]), duration=0.4, method=InterpolationTechnique.MIN_JERK)
        except Exception as e:
            logger.debug(f"Error executing thinking: {e}")

    def _head_tilt_sync(self) -> None:
        """Synchronous head tilt movement (speaking)."""
        try:
            tilt = create_head_pose(z=random.randint(-15, 15), mm=True)
            self.reachy_mini.goto_target(head=tilt, duration=0.5, method=InterpolationTechnique.MIN_JERK)
        except Exception as e:
            logger.debug(f"Error executing head_tilt: {e}")

    def _antenna_wiggle_sync(self) -> None:
        """Synchronous antenna wiggle movement (speaking)."""
        try:
            self.reachy_mini.goto_target(antennas=np.deg2rad([40, -40]), duration=0.4, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.4)
            self.reachy_mini.goto_target(antennas=np.deg2rad([-40, 40]), duration=0.4, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.4)
            self.reachy_mini.goto_target(antennas=np.deg2rad([0, 0]), duration=0.3, method=InterpolationTechnique.MIN_JERK)
        except Exception as e:
            logger.debug(f"Error executing antenna_wiggle: {e}")

    def _antenna_wave_sync(self) -> None:
        """Synchronous antenna wave movement (speaking)."""
        try:
            self.reachy_mini.goto_target(antennas=np.deg2rad([50, 50]), duration=0.5, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.5)
            self.reachy_mini.goto_target(antennas=np.deg2rad([25, 25]), duration=0.4, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.4)
            self.reachy_mini.goto_target(antennas=np.deg2rad([0, 0]), duration=0.3, method=InterpolationTechnique.MIN_JERK)
        except Exception as e:
            logger.debug(f"Error executing antenna_wave: {e}")

    def _head_bob_sync(self) -> None:
        """Synchronous head bob movement (speaking)."""
        try:
            up = create_head_pose(z=12, mm=True)
            neutral = create_head_pose(z=0, mm=True)
            self.reachy_mini.goto_target(head=up, duration=0.35, method=InterpolationTechnique.MIN_JERK)
            time.sleep(0.35)
            self.reachy_mini.goto_target(head=neutral, duration=0.35, method=InterpolationTechnique.MIN_JERK)
        except Exception as e:
            logger.debug(f"Error executing head_bob: {e}")
