"""
CANCELLATION
Long-running work (extraction, probes, bootstrap intervals) takes callbacks; raising
Cancelled from one of them stops the work at that point. Nothing here touches Tk.
"""


class Cancelled(Exception):
    """The user asked to stop. Raised by a progress or check callback; the work cleans up and re-raises it."""
