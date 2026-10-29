# RecipeParse: Algorithm Explanations and Pseudocode

This document details the mathematical formulations, algorithmic logic, and pseudocode for all algorithms implemented in RecipeParse.

---

## 1. Rule-Based Marathi Stemmer & Lemmatizer

### Linguistic Context
Marathi is a fusional and agglutinative Indo-Aryan language. Nouns and verbs undergo extensive inflection with postpositions and case markers. The English Porter Stemmer is completely ineffective for Devanagari.

### Stemmer Pseudocode
```text
Algorithm: MarathiSuffixStrippingStemmer(word, suffix_rules)
Input: A Marathi token 'word', sorted list of Marathi suffixes 'suffix_rules'
Output: Stemmed token

1. If length(word) <= 3:
       Return word
2. For each suffix in suffix_rules (sorted descending by length):
       If word ends with suffix:
           stem = word without suffix
           If length(stem) >= 2:
               Return stem
3. Return word
```

### Lemmatizer Pseudocode
```text
Algorithm: MarathiDomainLemmatizer(word, lemma_dict)
Input: A Marathi token 'word', domain dictionary 'lemma_dict'
Output: Base lemma

1. normalized_word = Unicode_NFC(word)
2. If normalized_word in lemma_dict:
       Return lemma_dict[normalized_word]
3. Else:
       Return MarathiSuffixStrippingStemmer(normalized_word)
```

---

## 2. N-Gram Language Model with Add-k Smoothing

### Formulation
For a sequence of words $w_1, w_2, \dots, w_n$:
- **Unigram Probability**:
  $$P(w) = \frac{C(w) + k}{N + k \cdot |V|}$$
- **Bigram Probability**:
  $$P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + k \cdot |V|}$$
- **Trigram Probability**:
  $$P(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i) + k}{C(w_{i-2}, w_{i-1}) + k \cdot |V|}$$

### Backoff Prediction Logic
```text
Algorithm: PredictNextWordTrigram(w1, w2, top_k)
Input: Context tokens w1, w2; top_k count
Output: List of predicted candidate words with probabilities

1. If (w1, w2) exists in Bigram counts:
       Retrieve all candidate words w3 where Trigram(w1, w2, w3) > 0
       Compute P(w3 | w1, w2) using Add-k smoothing
2. Else:
       Backoff to Bigram: Retrieve candidate words w3 where Bigram(w2, w3) > 0
       Compute P(w3 | w2)
3. If still empty:
       Fallback to top frequent Unigrams in vocabulary
4. Sort candidates by probability descending and return top_k
```

---

## 3. Hidden Markov Model (HMM) & Viterbi Decoding

### Mathematical Formulation
Given observation sequence $O = (o_1, o_2, \dots, o_T)$ and state sequence $S = (s_1, s_2, \dots, s_T)$ from tagset $\Sigma$:
- **Initial Probability**: $\pi_i = P(s_1 = i)$
- **Transition Probability**: $A_{ij} = P(s_t = j \mid s_{t-1} = i)$
- **Emission Probability**: $B_j(o) = P(o_t = o \mid s_t = j)$

### Viterbi Dynamic Programming Algorithm
```text
Algorithm: ViterbiDecoding(tokens, states, Pi, A, B)
Input: Observation sequence tokens O = (o_1, ..., o_T), states Sigma
Output: Best state sequence S*

1. Initialize trellis table V[T, |Sigma|] and backpointer table BP[T, |Sigma|]
2. Initialization step (t = 0):
       For each state s in Sigma:
           V[0, s] = log(Pi[s]) + log(B[s, o_0])
           BP[0, s] = NULL

3. Recursion step (t = 1 to T-1):
       For each state s in Sigma:
           V[t, s] = max_{s'} [ V[t-1, s'] + log(A[s', s]) ] + log(B[s, o_t])
           BP[t, s] = argmax_{s'} [ V[t-1, s'] + log(A[s', s]) ]

4. Termination step:
       s_T* = argmax_{s} V[T-1, s]

5. Backtracking step (t = T-1 down to 1):
       s_{t-1}* = BP[t, s_t*]

6. Return sequence (s_0*, s_1*, ..., s_{T-1}*)
```

---

## 4. Simplified Lesk Overlap Algorithm for Word Sense Disambiguation (WSD)

### Formulation
For an ambiguous Marathi target word $w$ occurring in context sentence $C$:
$$\text{BestSense} = \arg\max_{S_i \in \text{Senses}(w)} \left| \text{ContextWords}(C) \cap \text{SignatureWords}(S_i) \right|$$

### Algorithm Pseudocode
```text
Algorithm: SimplifiedLeskMarathi(sentence, target_word, lexicon)
Input: Context sentence, polysemous target_word, lexicon
Output: Selected sense ID and explanation

1. context_words = Tokenize(sentence) - Stopwords - {target_word}
2. candidate_senses = lexicon[target_word].senses
3. max_overlap = -1, best_sense = candidate_senses[0]

4. For each sense S in candidate_senses:
       signature = Tokenize(S.gloss) + Tokenize(S.examples) - Stopwords - {target_word}
       overlap = context_words INTERSECT signature
       score = |overlap|
       If score > max_overlap:
           max_overlap = score
           best_sense = S

5. Return best_sense, max_overlap, and overlapping words
```
