"""Tweet-vs-post vocabulary switch (the `FC` singleton) and the path of the live prompts JSON."""
import os


class _FormatConfig:
    """Repo-wide switch between "post" and "tweet" wording, plus the live prompts file path.

    Default mode is "post" (the prompt JSONs ask for `POST: <content>`); set
    ABM_FORMAT=tweet for the old tweet wording. Import the `FC` singleton and read its
    attributes, e.g. `FC.NO_CONTENT` ("NO_POST" or "NO_TWEET") or `FC.PROMPTS_FILE`.
    """
    def __init__(self):
        """Read ABM_FORMAT and set the vocabulary, sentinel and path attributes."""
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
        self.PROMPTS_FILE = "data/prompts_optimal.json"

        # Suffix appended to data / plot base directories
        self.DIR_SUFFIX = "_post" if self.is_post else ""


FC = _FormatConfig()
