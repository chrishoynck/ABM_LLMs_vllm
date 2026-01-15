import numpy as np
import utils.metrics as metrics
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
        # self.identity: str = identity
        self.agent_connections = set()
        self.activation_state = False
        self._next_last_tweet: str  = "NO_TWEET"
        # self.response_threshold = rng.random() if rng else np.random.random()
        self.persona = persona
        self.well_being = well_being
        self.phq9_score = well_being.get("phq9_sumscore") if well_being else None
        self.age = well_being.get("age") if well_being else None

        # Additional attributes for LLM interaction
        # self.rng = rng if rng else np.random.default_rng()

        self._force_active = False
        self.tweethistory = []
        

        self.last_tweet: str | None = None
        self._next_activation_state = False 

        self.distorted_tweets = []
        self.active_tweethistory = []
        self.frac_distorted_neigh = 0

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

    def persona_prompt(self):
        if self.persona is None:
            return "You have no specific persona."
        p = self.persona

        hobbies = ", ".join(p["hobbies"][:5]) if p["hobbies"] else "no particular hobbies"
        skills = ", ".join(p["skills"][:5]) if p["skills"] else "no specific skills"

        # combine the free-text persona + structured info
        base = f"You are {p['name']} " #, {p['persona_text'].rstrip()}. "
        extra = (
            f"You are {p['age']} years old, gender: {p['sex']},"
            f"Marital status: {p['marital_status']}, living in {p['city']}. "
            f"worklife: {p['occupation'].replace('_', ' ')}. "
            f"your hobbies include {hobbies}, and your key skills are {skills}."
        )
        return base + extra
    
    def well_being_prompt(self,well_being : dict):

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
            f"Current well-being: PHQ-9 score {score:.4f} "
            f"({severity} depression). The person {dep_flag} for clinically depression"
            # f"relevant depressive symptoms and {diag_flag}.{extra_txt}"
        )


    def build_tweet_prompt(self, tokenizer, round_idx, neighbor_pairs, max_chars=240, force_active=False):
        # neighbor_pairs: list of (neighbor_id, last_text)
        # own history block
        own_block = "" 
        if len(self.tweethistory) == 0:
            own_block = "(no own previous tweets)"
        else:
            recent = list(reversed(self.tweethistory[-2:]))  # newest first
            own_block = "\n".join(f"- {t[:max_chars]}" for t in recent)
        
        # neighbor_pairs = self.rng.permutation(neighbor_pairs)  # shuffle neighbor tweets
        neighbor_block = "(no neighbor tweets)" if len(neighbor_pairs) == 0 else "\n".join(
            f"- Agent {nid}: {txt[:max_chars]}" for nid, txt in neighbor_pairs[:5]  # limit to first 5 neighbors
        )

        # agents are asked to tweet
        if force_active:
            system = (f"You are a social media user {self.ID}.\n"
                       "Think of a hobby that fits your age, and an interesting short tweet to post.\n"
                       "You must post a short tweet (<= " f"{max_chars} chars).\n"
                       "You are given well-being information about yourself, your tone can be influenced by it, but the content may not.\n\n"
                       "POST FORMAT (exactly):\n"
                    #    "Start your reply with: TWEET:\n"
                    #    "Then include your tweet text on the next line:\n"
                       "Reply with \"TWEET: <your tweet text>\"\n"
                       "Do not add anything else, do not explain.\n\n"
                       )
            user = (
                f"Round: {round_idx}\n"
                # f"Identity: {self.persona_prompt()}\n"
                f"Well-being: {self.well_being_prompt(self.well_being)}\n"
            )
            
        # The agents may decide not to tweet
        else:
            system = (
            f"You are a social media user {self.ID}.\n"
            "Your task: Read the provided neighbor tweets and decide whether to post a new tweet yourself.\n\n"
            "You are given well-being information about yourself, your tone can be influenced by it, but the content may not.\n\n"
            "### INSTRUCTIONS ###\n"
            f"1. Read the neigbor tweets and think of an tweet to post yourself\n"
            f"2. Make a decision: to tweet or NOT to tweet.\n"
            f"3. If you decide to tweet, the tweet must start with \"TWEET\" and be <= {max_chars} characters.\n\n"
            f"4. If you decide NOT to tweet, simply reply with \"NO_TWEET\".\n\n"
            f"5. Do not explain, just reply in the specified format.\n\n"
            
            "### OUTPUT FORMAT ###\n"
            "You must reply using ONLY one of the following two formats (do not add explanations):\n\n"
            
            "[If you decide to tweet]:\n"
            "TWEET: <write your tweet content here>\n\n"
            
            "[If you decide NOT to tweet]:\n"
            "NO_TWEET\n"
        )
            user = (
                f"Round: {round_idx}\n"
                # f"Identity: {self.persona_prompt()}\n"
                f"Well-being: {self.well_being_prompt(self.well_being)}\n"
                f"Neighbor tweets:\n{neighbor_block}\n"
                f"Your recent previous tweets:\n{own_block}\n"
            )

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # print("PROMPT MESSAGES: ", messages)
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def step_llm_tweet(self, tokenizer, rng, round_idx:int, max_chars = 240, force_active=False):
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
                if n.distorted_tweets[-1]:
                    distorted_neigh +=1
        
        if len(activated_neighbors) > 0:
            self.frac_distorted_neigh = distorted_neigh/len(activated_neighbors)
        else:
            self.frac_distorted_neigh = 0
            
        neighbor_msgs = rng.permutation(neighbor_msgs)[:5]  # limit to first 5 neighbors

        # force tweet if needed
        self._force_active = force_active

        # create prompt
        prompt = self.build_tweet_prompt(
            tokenizer, round_idx, neighbor_msgs, max_chars=max_chars, force_active=force_active
        )

        return prompt
    
    def send_tweet(self, max_chars, raw_tweet):
        '''
        Process the raw tweet output from the LLM and update the agent's next tweet and activation  state.
        Args:
            max_chars (int): The maximum number of characters for the tweet.
            raw_tweet (str): The raw tweet output from the LLM.
        '''
        if self._force_active:
            print (f"Agent {self.ID} FORCED TWEET OUTPUT: {raw_tweet}")

        do_tweet, tweet = self.parse_tweet_decision(raw_tweet)
        if do_tweet:
            # prepare tweet and set next activations
            tweet = tweet.strip()
            if len(tweet) > max_chars:
                tweet = tweet[:max_chars]
            self._next_last_tweet = tweet
            self._next_activation_state = True

        else:
            # if formatted incorrectly or NO_TWEET, send out NO_TWEET
            self._next_last_tweet = "NO_TWEET"
            self._next_activation_state = False
        
    # Finalize the activation state for this step
    def commit(self, n_grams):
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
        return distorted

    def reset_activation_state(self):
        '''
        Reset the activation state of the agent.
        '''
        self.activation_state = False

    def parse_tweet_decision(self, text: str):
        """
        Parse the LLM output to determine if the agent decided to tweet or not.
        Args:
            text (str): The LLM output text.    
        Returns:
            bool: True if the agent decided to tweet, False otherwise.
            str: The tweet text if the agent decided to tweet, empty string otherwise.
        """
        t = text.strip()
        low = t.lower()
        low = low.replace("\\'", "'")
        low = low.replace("\\n", " ")

        # all text that is not generated in proper format is treated as no tweet
        if "no_tweet" in low:
            return False, ""
        if "tweet:" not in low:
            return False, ""
        # prefer the explicit "TWEET:" pattern
        idx = low.find("tweet:")
        if idx != -1:
    
            return True, t[idx:].strip()
        
        # fallback: if any non-empty content, treat as tweet
        return (len(t) > 0), t
        
    def respond(self) -> set:
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
        return hash((self.ID, self.persona['name'] if self.persona else None)) 

    def __eq__(self, other):
        """
        Check if the agent is equal to another agent.
        """
        return isinstance(other, Agent) and self.ID == other.ID 

