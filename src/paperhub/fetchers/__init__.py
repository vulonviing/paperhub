"""HuggingFace Daily Papers fetchers."""

from .base import Fetcher
from .hf_api import HFAPIFetcher
from .hf_html import HFHtmlFetcher
from .range import papers_in_range

__all__ = ["Fetcher", "HFAPIFetcher", "HFHtmlFetcher", "papers_in_range"]
