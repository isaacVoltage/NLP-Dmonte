"""
Experiment 6: Viterbi Algorithm Implementation for HMM POS Tagging
Decodes the most probable hidden POS state sequence given an observation sequence
using dynamic programming:
viterbi[t, state] = max_{prev_state} (viterbi[t-1, prev_state] * P(state|prev_state) * P(observation_t|state))
backpointer[t, state] = argmax_{prev_state} (...)
"""
import math
from typing import List, Dict, Tuple, Any

class ViterbiDecoder:
    def __init__(self, states: List[str], initial_probs: Dict[str, float],
                 transition_probs: Dict[str, Dict[str, float]],
                 emission_probs: Dict[str, Dict[str, float]],
                 unknown_emission_prob: float = 1e-5):
        self.states = sorted(list(set(states)))
        self.initial_probs = initial_probs
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs
        self.unknown_emission_prob = unknown_emission_prob

    def decode(self, tokens: List[str]) -> Tuple[List[Tuple[str, str, float]], List[Dict[str, Any]]]:
        """
        Runs Viterbi dynamic programming decoding over observation tokens.
        Returns:
            - List of (token, predicted_tag, state_probability)
            - Step-by-step trellis table for educational inspection
        """
        if not tokens:
            return [], []

        T = len(tokens)
        N = len(self.states)

        # viterbi[t][s] holds the max log probability of path ending at state s at step t
        viterbi = [{} for _ in range(T)]
        backpointer = [{} for _ in range(T)]

        # Trellis log records for educational display
        trellis_steps = []

        # 1. Initialization (t = 0)
        first_token = tokens[0]
        step_0_record = {"step": 1, "token": first_token, "state_scores": {}}

        for s in self.states:
            init_p = self.initial_probs.get(s, 1e-6)
            emit_p = self.emission_probs.get(s, {}).get(first_token, self.unknown_emission_prob)
            
            # Log probabilities to avoid floating point underflow
            prob = math.log(max(init_p, 1e-12)) + math.log(max(emit_p, 1e-12))
            viterbi[0][s] = prob
            backpointer[0][s] = None
            step_0_record["state_scores"][s] = round(prob, 3)

        trellis_steps.append(step_0_record)

        # 2. Recursion (t = 1 to T-1)
        for t in range(1, T):
            token = tokens[t]
            step_record = {"step": t + 1, "token": token, "state_scores": {}}

            for s in self.states:
                emit_p = self.emission_probs.get(s, {}).get(token, self.unknown_emission_prob)
                log_emit = math.log(max(emit_p, 1e-12))

                # Find max over all previous states s_prev
                best_prev_state = None
                best_prob = -float('inf')

                for s_prev in self.states:
                    trans_p = self.transition_probs.get(s_prev, {}).get(s, 1e-6)
                    log_trans = math.log(max(trans_p, 1e-12))
                    total_p = viterbi[t-1][s_prev] + log_trans + log_emit

                    if total_p > best_prob:
                        best_prob = total_p
                        best_prev_state = s_prev

                viterbi[t][s] = best_prob
                backpointer[t][s] = best_prev_state
                step_record["state_scores"][s] = round(best_prob, 3)

            trellis_steps.append(step_record)

        # 3. Termination: find best final state
        best_last_state = max(self.states, key=lambda s: viterbi[T-1][s])
        best_final_prob = viterbi[T-1][best_last_state]

        # 4. Path Backtracking
        best_path = [best_last_state]
        for t in range(T - 1, 0, -1):
            prev_state = backpointer[t][best_path[-1]]
            best_path.append(prev_state)
        best_path.reverse()

        # Format tagged output with relative confidence score
        results = []
        for t in range(T):
            tag = best_path[t]
            # Convert log score back to normalized relative confidence
            all_scores = [viterbi[t][s] for s in self.states]
            max_s = max(all_scores)
            sum_exp = sum(math.exp(sc - max_s) for sc in all_scores)
            tag_conf = math.exp(viterbi[t][tag] - max_s) / sum_exp if sum_exp > 0 else 1.0

            results.append((tokens[t], tag, round(tag_conf, 4)))

        return results, trellis_steps
