import os


class _FormatConfig:
    """
    Central configuration for tweet vs. post format mode.

    Set the environment variable  ABM_FORMAT=post  to switch the entire
    simulation from "tweet" terminology to "post" terminology.
    Default (unset or "tweet") keeps the original tweet behaviour.

    Usage in other modules:
        from utils.format_config import FC
        ...
        if raw == FC.NO_CONTENT:   # "NO_TWEET" or "NO_POST"
    """
    def __init__(self):
        mode = os.environ.get("ABM_FORMAT", "tweet").strip().lower()
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
        self.PROMPTS_FILE = (
            "data/prompts_post.json" if self.is_post else "data/prompts.json"
        )

        # Suffix appended to data / plot base directories
        self.DIR_SUFFIX = "_post" if self.is_post else ""


FC = _FormatConfig()
