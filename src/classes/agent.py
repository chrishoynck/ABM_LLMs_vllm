import numpy as np
import utils.metrics as metrics
from utils.format_config import FC
import re
import json

class Agent:
    """
    A agent in the network, with a unique ID and a response threshold.
    The response threshold is a random number between 0 and 1, which is used to determine whether the agent will respond to a piece of news.
    The agent can be in one of two states: activated or not activated.
    The agent can also be a sampler, which means that it will always respond to a piece of news, regardless of the response threshold.
    """
    def __init__(self, ID, rng=None, persona=None, well_being=None):
        """
        Initialize the agent.

        Args:
            ID (int): The unique ID of the agent.
            rng (np.random.Generator, optional): The random number generator to use. Defaults to None.
        
        Attributes:
            response_threshold (float): The response threshold of the agent.
            activation_state (bool): Whether the agent is activated or not.
            agent_connections (set): The set of agents that the agent is connected to.
        """
        self.ID = ID
        self.agent_connections = set()
        self.activation_state = False
        self._next_last_tweet: str  = FC.NO_CONTENT
        self.persona = persona
        self.well_being = well_being
        self.phq9_score = well_being.get("phq9_sumscore") if well_being else None
        self.age = well_being.get("age") if well_being else None
        self.all_phq9_sumscores = []

        with open(FC.PROMPTS_FILE, 'r') as f:
           self._PROMPTS = json.load(f)

        self._force_active = False
        self.tweethistory = []
        
        self.last_tweet: str | None = None
        self._next_activation_state = False 

        self.distorted_tweets = []
        self.active_tweethistory = []
        self.frac_distorted_neigh = 0
        self._tweets_since_phq9_update = 0  # counter for efficient same-score tweet lookup

    @staticmethod
    def phq9_severity_category(score: float) -> str:
        """Map PHQ-9 sumscore to a standard severity label."""
        try:
            s = float(score)
        except (TypeError, ValueError):
            return "unknown"
        if s <= 4:
            return "none/minimal"
        elif s <= 9:
            return "mild"
        elif s <= 14:
            return "moderate"
        elif s <= 19:
            return "moderately severe"
        else:
            return "severe"
    
    _REASONING_LINE_RE = re.compile(
        r'^\s*('
        r'\d+\.\s+\*\*'                  # "7.  **Final Polish:**"
        r'|\*[*\s]'                       # asterisk + asterisk/space (reasoning marker)
        r'|Wait\s*,'                      # "Wait, I need…"
        r'|-\s+\*'                        # "-   *Refining…"
        r'|id\s*:\s*\d'                   # "ID: 1"
        r'|persona\s*:'                   # "Persona: …"
        r')',
        re.IGNORECASE,
    )

    @staticmethod
    def strip_model_thinking(text: str) -> str:
        """Strip chain-of-thought / reasoning blocks that some models
        (e.g. Qwen3.5) produce even with thinking enabled/disabled.

        Handles (in order):
        1. Full <think>...</think> XML blocks (Qwen3 format)
        2. Qwen3.5 format: <think> is placed in the *prompt* by the chat
           template, so only </think> appears in the generated output.
           Everything before (and including) the first </think> is reasoning.
        3. Unclosed <think> blocks (model started but never closed)
        4. Plain-text "Thinking Process:" blocks
        """
        # 1. Remove complete <think>...</think> XML blocks
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # 2. Qwen3.5 format: only </think> in the output (opening tag was in prompt).
        #    Strip everything up to and including the first </think>.
        if '</think>' in cleaned and '<think>' not in cleaned:
            cleaned = cleaned.split('</think>', 1)[1].strip()

        # 3. Handle unclosed <think> blocks
        if '<think>' in cleaned:
            cleaned = cleaned[:cleaned.index('<think>')].strip()

        # 4. If the model produced plain-text reasoning blocks, keep only the
        #    text after the last one (the actual answer).
        parts = re.split(r'(?i)\bThinking Process\s*:', cleaned)
        if len(parts) > 1:
            cleaned = parts[-1].strip()

        return cleaned

    @staticmethod
    def parse_phq9_answers(answers: str) -> int:
        """Parse the PHQ-9 answers from the LLM output and compute the sumscore.

        Only considers lines whose left-hand side matches a PHQ-9 question
        label (Q1 … Q9) to avoid picking up stray digits from reasoning text.
        """
        cleaned = Agent.strip_model_thinking(answers)
        lines = cleaned.strip().split("\n")
        total_score = 0

        for line in lines:
            stripped = line.strip()
            match = re.match(r'^Q\s*(\d+)\s*:\s*(\d)', stripped, re.IGNORECASE)
            if not match:
                continue
            q_num = int(match.group(1))
            if q_num < 1 or q_num > 9:
                continue
            score = int(match.group(2))
            if 0 <= score <= 3:
                total_score += score
            else:
                print(f"Score out of range (found {score}) in line: {line}")

        return total_score

    
    # def persona_prompt(self):
    #     if self.persona is None:
    #         return "You have no specific persona."
    #     p = self.persona

    #     hobbies = ", ".join(p["hobbies"][:5]) if p["hobbies"] else "no particular hobbies"
    #     skills = ", ".join(p["skills"][:5]) if p["skills"] else "no specific skills"

    #     # combine the free-text persona + structured info
    #     base = f"You are {p['name']} " #, {p['persona_text'].rstrip()}. "
    #     extra = (
    #         f"You are {p['age']} years old, gender: {p['sex']},"
    #         f"Marital status: {p['marital_status']}, living in {p['city']}. "
    #         f"worklife: {p['occupation'].replace('_', ' ')}. "
    #         f"your hobbies include {hobbies}, and your key skills are {skills}."
    #     )
    #     return base + extra
    
    @staticmethod
    def well_being_prompt(well_being : dict):

        """
        Build a concise well-being prompt based on PHQ-9 and related fields.

        Expects `well_being` to be the output of `parse_phq9`.
        """
        score = well_being.get("phq9_sumscore")
        severity = Agent.phq9_severity_category(score)

        dep_symp = well_being.get("depressive_symptoms")
        diagnosis = well_being.get("diagnosis")
        # freq_eps = well_being.get("Freq_depressive_episodes")
        # age_first = well_being.get("Age_first_depressive_episode")

        # Short flags
        dep_flag = "screens positive" if dep_symp else "does not screen positive"
        diag_flag = "has a history of MDD" if diagnosis else "has no recorded MDD diagnosis"

        extra_bits = []
        # if freq_eps is not None:
        #     extra_bits.append(f"reported frequency of depressive episodes: {freq_eps}")
        # if age_first is not None:
        #     extra_bits.append(f"first episode around age {age_first}")

        extra_txt = ". " + "; ".join(extra_bits) if extra_bits else ""

        return (
            f"You are {well_being['age']} years old."
            f"Current well-being: PHQ-9 score {score:.4f} ({severity} depression) "
            # f"({severity} depression). The person {dep_flag} for clinically depression"
            # f"relevant depressive symptoms and {diag_flag}.{extra_txt}"
        )
    
    def phq9_questionnaire_prompt(self, tokenizer, tweets: list[str], force_active=False, use_persona=False):
        """
        build a phq9_questionnaire_prompt
        """
        if self.well_being is None:
            well_being_info =  "No well-being information available."
        else:
            well_being_info = self.well_being_prompt(self.well_being)
        
        system = self._PROMPTS["phq9"]["system_user"]
        
        # involve persona in the prompt
        if use_persona:
            system = self._PROMPTS["phq9"]["system_persona"]
            user = self._PROMPTS["phq9"]["user_template_persona"].format(
                agent_id=self.ID,
                persona=self.persona,
                tweets_block="\n".join(tweets)
            )
        
        # do not provide well-being information 
        elif force_active:
            user = self._PROMPTS["phq9"]["user_template_forced"].format(
                agent_id=self.ID,
                tweets_block="\n".join(tweets)
            )
        else:
            user = self._PROMPTS["phq9"]["user_template_user"].format(
                agent_id=self.ID,
                well_being_info=well_being_info,
                tweets_block="\n".join(tweets)
            )

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # print("PROMPT MESSAGES: ", messages)
        return messages
        # return tokenizer.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )


    def build_tweet_prompt(self, tokenizer, round_idx, neighbor_pairs, max_chars=240, force_active=False, tweet_block_phq9=False):

        # own history block
        own_block = "" 
        if len(self.tweethistory) == 0:
            own_block = f"(no own previous {FC.label_plural})"
        else:
            recent = list(reversed(self.tweethistory[-1:]))  # newest first
            own_block = "\n(do not repeat)".join(f"- {t[:max_chars]}" for t in recent)
        
        neighbor_block = f"(no neighbor {FC.label_plural})" if len(neighbor_pairs) == 0 else "\n".join(
            f"- Agent {nid}: {txt[:max_chars]}" for nid, txt in neighbor_pairs 
        )

        prompt_cfg = self._PROMPTS["tweet_gen"]
        system_str = prompt_cfg["system_forced"] if force_active else prompt_cfg["system_standard"]
        
        system_content = system_str.format(max_chars=max_chars)
        if not force_active:
            if tweet_block_phq9 and self._tweets_since_phq9_update > 0:
                recent = self.tweethistory[-self._tweets_since_phq9_update:]
                same_score = [t for t in recent if t and t != FC.NO_CONTENT]
                if same_score:
                    items = "\n".join(f'- "{t[:max_chars]}"' for t in same_score)
                    own_block = (
                        f"\nYour previous {FC.label_plural} for this well-being state:\n{items}\n"
                        f"(Do NOT adopt the same topics or reuse the same words. "
                        f"Be original and vary your content.)"
                    )
                else:
                    own_block = ""

            user_content = prompt_cfg["user_template"].format(
                agent_id = self.ID,
                persona=self.persona,
                well_being=self.well_being_prompt(self.well_being),
                neighbor_block=neighbor_block,
                own_block=own_block
            )
        else:
            if tweet_block_phq9 and self._tweets_since_phq9_update > 0:
                recent = self.tweethistory[-self._tweets_since_phq9_update:]
                same_score = [t for t in recent if t and t != FC.NO_CONTENT]
                if same_score:
                    items = "\n".join(f'- "{t[:max_chars]}"' for t in same_score)
                    previous_tweet_block = (
                        f"\nYour previous {FC.label_plural} for this well-being state:\n{items}\n"
                        f"(Do NOT adopt the same topics or reuse the same words. "
                        f"Be original and vary your content.)"
                    )
                else:
                    previous_tweet_block = ""
            elif self.last_tweet and self.last_tweet != FC.NO_CONTENT:
                previous_tweet_block = (
                    f"\nYour previous {FC.label}: \"{self.last_tweet[:max_chars]}\"\n"
                    f"(Do NOT adopt the same topic or reuse the same words. "
                    f"Be original and vary your content.)"
                )
            else:
                previous_tweet_block = ""

            user_content = prompt_cfg["user_template_forced"].format(
                agent_id = self.ID,
                persona=self.persona,
                well_being=self.well_being_prompt(self.well_being),
                previous_tweet_block=previous_tweet_block,
            )

        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
        
        # print("PROMPT MESSAGES: ", messages)
        return messages
        # return tokenizer.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
    
    def step_llm_tweet(self, tokenizer, rng, round_idx, max_chars=240, force_active=False, tweet_block_phq9=False):
        """
        Use the LLM to decide whether to tweet or not.

        Args:
            round_idx (int): The current round index.
            max_chars (int, optional): The maximum number of characters for the tweet. Defaults to 240.
        Returns:
            bool: Whether the agent decided to tweet or not.
        """
        neighbor_msgs = []
        activated_neighbors = self.respond()
        distorted_neigh = 0

        # gather neighbor tweets
        for n in activated_neighbors:
            if n.activation_state and n.last_tweet:
                neighbor_msgs.append((n.ID, n.last_tweet))
                if len(n.distorted_tweets) > 0 and n.distorted_tweets[-1]:
                    distorted_neigh +=1
        
        if len(activated_neighbors) > 0:
            self.frac_distorted_neigh = distorted_neigh/len(activated_neighbors)
        else:
            self.frac_distorted_neigh = 0
            
        neighbor_msgs = rng.permutation(neighbor_msgs)[:5]  # limit to first 10 neighbors

        # force tweet if needed
        self._force_active = force_active

        # create prompt
        prompt = self.build_tweet_prompt(
            tokenizer, round_idx, neighbor_msgs, max_chars=max_chars,
            force_active=force_active, tweet_block_phq9=tweet_block_phq9
        )

        return prompt
    
    def send_tweet(self, max_chars, raw_tweet):
        '''
        Process the raw tweet output from the LLM and update the agent's next tweet and activation  state.
        Args:
            max_chars (int): The maximum number of characters for the tweet.
            raw_tweet (str): The raw tweet output from the LLM.
        '''
        # if self._force_active:
            # print (f"Agent {self.ID} FORCED TWEET OUTPUT: {raw_tweet}")

        do_tweet, tweet = self.parse_tweet_decision(raw_tweet)
        if do_tweet:
            # prepare tweet and set next activations
            tweet = tweet.strip()
            if len(tweet) > max_chars:
                tweet = tweet[:max_chars]
            self._next_last_tweet = tweet
            self._next_activation_state = True

        else:
            self._next_last_tweet = FC.NO_CONTENT
            self._next_activation_state = False
        
        print(f"Agent {self.ID}, POST/TWEET: {self._next_last_tweet}, phq-9: {self.well_being.get('phq9_sumscore') if self.well_being else 'None'}")
    # Finalize the activation state for this step
    def commit(self, n_grams, update_score=False) -> bool:
        """
        Commit the next activation state and last tweet.
        S.T all updates happen simultaneously after all agents have decided.
        """       
        tweetje = self._next_last_tweet
        distorted = False
        if  self._next_activation_state:

            # update distortion metrics
            distorted = metrics.contains_ngram(tweetje, ngrams=n_grams)
            self.distorted_tweets.append(distorted)
            self.distorted_tweets = self.distorted_tweets[-5:]
            self.active_tweethistory.append(tweetje)
            self.active_tweethistory = self.active_tweethistory[-5:]
    
        self.tweethistory.append(self._next_last_tweet)
        self.last_tweet = self._next_last_tweet
        self.activation_state = self._next_activation_state
        self._tweets_since_phq9_update += 1

        # record phq9 sumscore history (may be updated)
        if True: #update_score:
            self.all_phq9_sumscores.append(self.well_being.get("phq9_sumscore") if self.well_being else None)

        return distorted
    
    def update_well_being(self, sumscore: int, new_phq9=False):
        """
        Update the well-being information of the agent.

        Args:
            sumscore (int): The new PHQ-9 sumscore.
        """

        print(f"Agent {self.ID} PHQ-9 sumscore updated to {sumscore} (old PHQ-9 sumscore: {self.well_being.get('phq9_sumscore') if self.well_being else 'None'}).")
        if self.well_being is None:
            self.well_being = {}
        self.well_being["phq9_sumscore"] = sumscore
        if new_phq9:
            self._tweets_since_phq9_update = 0

        # update diagnosis flag???

        # record history
        # self.all_phq9_sumscores.append(sumscore)

    def reset_activation_state(self):
        '''
        Reset the activation state of the agent.
        '''
        self.activation_state = False

    def parse_tweet_decision(self, text: str):
        """
        Parse the LLM output to determine if the agent decided to post/tweet.
        Recognises both TWEET:/NO_TWEET and POST:/NO_POST based on ABM_FORMAT.

        Strategy:
          1. Strip thinking, extract tweet from the clean answer.
          2. If the answer is a ``<content>`` placeholder (model ran out of
             tokens before writing the real tweet), fall back to searching the
             *full* output (including the thinking block) for the last POST:
             with real content.
          3. Last resort: grab the last long quoted string from the thinking
             block (models typically put their final draft in quotes).
        """
        cleaned = self.strip_model_thinking(text)

        # -- primary path: clean answer after thinking is stripped ----------
        low_clean = cleaned.lower().replace("\\'", "'").replace("\\n", " ")

        no_kw  = FC.NO_CONTENT_LOWER          # "no_tweet" or "no_post"
        prefix_kw = FC.CONTENT_PREFIX_LOWER    # "tweet:" or "post:"

        last_no   = low_clean.rfind(no_kw)
        last_post = low_clean.rfind(prefix_kw)

        if last_no > last_post:
            return False, ""

        if last_post != -1:
            tweet = self._extract_content_at(cleaned, last_post + len(prefix_kw))
            if tweet and not self._is_placeholder(tweet):
                return True, tweet

        # -- fallback 1: scan full output for the last real POST: -----------
        tweet = self._find_last_real_tweet(text, prefix_kw)
        if tweet:
            return True, tweet

        # -- fallback 2: last long quoted string in the thinking block ------
        tweet = self._find_last_quoted_tweet(text)
        if tweet:
            return True, tweet

        return False, ""

    # ------------------------------------------------------------------
    #  Helpers used by parse_tweet_decision
    # ------------------------------------------------------------------

    @staticmethod
    def _is_placeholder(tweet: str) -> bool:
        """True when the tweet is a format template rather than real content."""
        low = tweet.lower()
        return '<content>' in low or '[content]' in low

    def _extract_content_at(self, text: str, content_start: int):
        """Extract and clean tweet content starting at *content_start*."""
        raw = text[content_start:].strip()

        first_para = re.split(r'\n\s*\n', raw, maxsplit=1)[0].strip()

        cleaned_lines = []
        for ln in first_para.split("\n"):
            if self._REASONING_LINE_RE.match(ln.strip()):
                break
            cleaned_lines.append(ln)

        tweet = "\n".join(cleaned_lines).strip().strip('`"\' ')

        if not tweet or tweet.lower().rstrip('.`\'" ') in ('...', '.', '', 'content'):
            return None
        return tweet

    @staticmethod
    def _find_last_real_tweet(text: str, prefix_kw: str):
        """Walk backwards through every POST:/TWEET: in *text* and return
        the last one whose content is not a ``<content>`` placeholder."""
        low = text.lower()
        search_end = len(low)
        while True:
            pos = low.rfind(prefix_kw, 0, search_end)
            if pos == -1:
                return None
            content_start = pos + len(prefix_kw)
            raw = text[content_start:].strip()
            first_para = re.split(r'\n\s*\n', raw, maxsplit=1)[0].strip()
            tweet = first_para.split("\n")[0].strip().strip('`"\' ')
            if tweet and '<content>' not in tweet.lower() \
                     and '[content]' not in tweet.lower() \
                     and tweet.lower().rstrip('.`\'" ') not in ('...', '.', '', 'content'):
                return tweet
            search_end = pos

    @staticmethod
    def _find_last_quoted_tweet(text: str, min_len: int = 50):
        """Extract the last long double-quoted string from *text*.
        Models typically put their final draft tweet in quotes during
        the thinking phase."""
        matches = re.findall(r'"([^"]{' + str(min_len) + r',})"', text)
        if not matches:
            return None
        candidate = matches[-1].strip()
        if '<content>' in candidate.lower():
            return None
        return candidate
        
    def respond(self) -> list:
        """
        Determine which connected agents are activated (sent out tweet).
        
        Returns:
            set: The set of agents that should be activated
        """
        actually_activated = []

        if len(self.agent_connections) > 0:
            actually_activated = [agent for agent in self.agent_connections if agent.activation_state] 
   
        # sort by ID for consistency
        return sorted(actually_activated, key=lambda a: a.ID)

    def add_edge(self, agent):
        """
        Add an edge to the agent.

        Args:
            agent (agent): The agent to add as an edge.
        """
        self.agent_connections.add(agent)

    def remove_edge(self, agent):
        """
        Remove an edge from the agent.

        Args:
            agent (agent): The Agent to remove as an edge.
        """
        self.agent_connections.discard(agent)
    
    def reset_agent(self):
        """
        Reset the agent to its initial state.
        """
        self.activation_state = False
        self.last_tweet = None
        self.tweethistory = []


    def __hash__(self):
        """
        Hash the agent by its ID and identity.
        Needed for the set data structure.

        Returns:
            int: The hash of the agent.
        """
        return hash((self.ID, self.persona if self.persona else None)) 

    def __eq__(self, other):
        """
        Check if the agent is equal to another agent.
        """
        return isinstance(other, Agent) and self.ID == other.ID 

