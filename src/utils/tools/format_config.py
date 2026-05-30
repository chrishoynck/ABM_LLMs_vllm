import os


class _FormatConfig:
    """
    Central configuration for tweet vs. post format mode.

    Default mode is "post" (matches data/prompts_post_minimal.json which asks
    the model for POST: <content>). Set ABM_FORMAT=tweet to revert to the
    legacy "tweet" terminology used by older prompt JSONs.

    Usage in other modules:
        from utils.tools.format_config import FC
        ...
        if raw == FC.NO_CONTENT:   # "NO_POST" or "NO_TWEET"
    """
    def __init__(self):
        mode = os.environ.get("ABM_FORMAT", "post").strip().lower()
        if mode not in ("tweet", "post"):
            raise ValueError(
                f"ABM_FORMAT must be 'tweet' or 'post', got '{mode}'"
            )
        self.mode = mode
        self.is_post = mode == "post"

        # Sentinel stored in agent history when there is no content
        self.NO_CONTENT = "NO_POST" if self.is_post else "NO_TWEET"

        # Prefix the LLM is asked to produce (and we parse for)
        self.CONTENT_PREFIX = "POST:" if self.is_post else "TWEET:"
        self.CONTENT_PREFIX_LOWER = self.CONTENT_PREFIX.lower()
        self.NO_CONTENT_LOWER = self.NO_CONTENT.lower().replace("_", "_")

        # Singular / plural labels used in hardcoded prompt fragments
        self.label = "post"  if self.is_post else "tweet"
        self.label_plural = "posts" if self.is_post else "tweets"
        self.Label = "Post"  if self.is_post else "Tweet"
        self.Label_plural = "Posts" if self.is_post else "Tweets"

        # Which prompt JSON to load
        self.PROMPTS_FILE = "data/prompts_post_minimal.json"

        # Suffix appended to data / plot base directories
        self.DIR_SUFFIX = "_post" if self.is_post else ""


FC = _FormatConfig()
