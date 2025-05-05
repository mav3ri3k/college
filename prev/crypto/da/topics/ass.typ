#import "lib.typ": *
#import "@preview/fletcher:0.5.5": *
#import "@preview/plotst:0.2.0": *

#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": chart


#let base = rgb("#eff1f5")

#show: doc => assignment(
  title: "Review of Generative Techniques in Deep Learning",

  course: "Cryptography and Network Security Lab",

  // Se apenas um author colocar , no final para indicar que é um array
  author: "Apurva Mishra: 22BCE2791",
  date: "3 March, 2025",
  doc,
)

= Abstract
Generative models have become central to advancements in deep learning, particularly with the rise of Large Language Models (LLMs) significantly influenced by the Transformer architecture. This paper presents a review of two dominant generative paradigms stemming from this architecture: autoregressive models and diffusion models. It begins by dissecting the foundational vanilla Transformer, detailing its encoder-decoder structure, attention mechanism, and layer normalization. The review then explores the principles of autoregressive generation, common in text-based LLMs, highlighting its sequential nature and associated challenges such as effective context length limitations and quadratic computational complexity. Subsequently, the paper introduces diffusion models, outlining the forward noising process and the learned backward denoising process, including considerations for training loss parameterization. Recognizing the inherent mismatch between continuous diffusion processes and discrete data like text, the review briefly surveys emerging techniques for discrete diffusion modeling, such as Mean Prediction, Ratio Matching, and Concrete Score Matching, culminating in the introduction of Score Entropy as a promising state-of-the-art approach. This review aims to provide a concise overview of the mechanisms, strengths, and challenges of these key generative techniques in modern deep learning.

= Intro

Large Language Models have been instrumental in recent advances in the
field of deep learning. This is in large part fueled through the introduction
of new transformer architecture for sequence modelling @vaswani2017attention.

Using this transformer architecture, two major paradigms have become
popular: Auto-regressive and diffusion based.

= Transformer
== Vanilla Transformer
The vanilla transformer @vaswani2017attention is a sequence to sequence model,
composed of a *encoder*, *stack of layers* and *decoder*.

*Encoder:*
The encoder is
responsible for converting the input sequence into representation in a
continuous latent space. It contains multi-head self attention mechanism
and position-wise feed-forward network (FFN). Finally each block is normalized
@ba2016layer.


  #show table.cell.where(y: 0): strong
  #set table(
    stroke: (x, y) => if y == 0 or y == 1 {
      (bottom: 0.7pt + black, top: 0.7pt + black)
    },
    align: (x, y) => (
      if x > 0 { center }
      else { left }
    )
  )

  #figure(
    table(
      columns: 3,
      table.header([*Algorithm:* Token Embedding]),

      [
        *Input*: $v #sym.in #sym.approx.eq [N_(V)]$, a token ID.\
        *Output*: $e #sym.in R^(d_(e))$ , the vector representation of the token.\
        *Parameters*: $W_(e) #sym.in R^(d_(e)*N_(V))$, the token embedding matrix.\

        *return* $e = W_(e)[:, v]$\
      ]
    ),
    caption: [
    Algorithm: Representing input tokens into latent space
    ]
  )


*Decoder:*
The decoder is responsible for converting the new transformation of the input
sequence in the latent space through the stack of layers back into text.
The decoder is similar to encoder in design. However it also includes a
additional multi-head attention layer between the self attention layer
and the feed-forward network (FFN) layer. Again this is also normalized.

  #figure(
    table(
      columns: 3,
      table.header([*Algorithm:* Un-Embedding]),

      [
        *Input*: $e #sym.in R^(d_e)$, a token encoding.\
        *Output*: $p #sym.in Delta (V)$ , a probability distribution over the vocabulary\
        *Parameters*: $W_(e) #sym.in R^(d_(e)*N_(V))$, the unembedding matrix.\

        *return* $"softmax"(W_u e)$\
      ]
    ),
    caption: [
    Algorithm: Converting vector from latent space to vocabulary
    ]
  )

*Layer Normalization*
For all: encoder, sub-layers and decoder layer normalization is performed for
the final output. This is essential as transformer have shown to become
unstable due to changes in activations from varying dataset. @berlyand2021stability
Here layer normalization ensures that each output is normalized to a consistent
distribution. This ensures a consistent distribution.

  #figure(
    table(
      columns: 3,
      table.header([*Algorithm:* Layer Normalization]),

      [
        *Input*: $e #sym.in R^(d_e)$, neural network activations.\
        *Output*: $e #sym.in R^(d_e)$, normalized activations.\
        *Parameters*: $y, #sym.beta R^(d_e)$, element-wise scale and offset\
        \
        1. $m #sym.arrow.l sum_(i=1)^(d_e) e[i]/d_e$
        2. $v #sym.arrow.l (sum_(i=1)^(d_e) (e[i]/d_e - m)^2)/d_e$

        *return* $e = (e-m)/sqrt(v) #sym.dot.circle y + #sym.beta$, where $#sym.dot.circle$ denotes element-wise multiplication.\
      ]
    ),
    caption: [
    Algorithm: Algorithm for layer normalization
    ]
  )
== Attention
The state of the art performance of vanilla transformer is obtained
using the attention mechanism. It works based on three inputs: _Query-Key-Value (QKV)_.
The function can be represented as:
$
  "Attention"(Q K V) = "softmax"((Q K^T)/(sqrt(D_(k))))V
$


#show table.cell.where(y: 0): strong
#set table(
  stroke: (x, y) => if y == 0 or y == 1 {
    (bottom: 0.7pt + black, top: 0.7pt + black)
  },
  align: (x, y) => (
    if x > 0 { center }
    else { left }
  )
)

#figure(
  table(
    columns: 3,
    table.header([*Algorithm:* Attention]),

    [
      *Input*: $v #sym.in #sym.approx.eq [N_(V)]$, a token ID.\
      *Output*: $e #sym.in R^(d_(e))$ , the vector representation of the token.\
      *Parameters*: $W_(e) #sym.in R^(d_(e)*N_(V))$, the token embedding matrix.\

      *return* $e = W_(e)[:, v]$\

      *Input*: $X #sym.in R^(d_(x)*l_(x)), Z #sym.in R^(d_(z)*l_(z))$, vector representation of primary and context sequence\
      *Output*: $V #sym.in R^(d_("out")*l_(x)),$, updated represetations of tokens in X, folding in information from token *Z*\ 

      *Parameter*: $W_(q k v)$ consist of\
                  $W_(q) #sym.in R^(d_("attn")*d_(x)), b_(q) #sym.in R^(d_("attn"))$\
                  $W_(k) #sym.in R^(d_("attn")*d_(x)), b_(k) #sym.in R^(d_("attn"))$\
                  $W_(v) #sym.in R^(d_("attn")*d_(x)), b_(v) #sym.in R^(d_("attn"))$\
      *Hyperparameters*: Mask $#sym.in {0, 1}^(l_(z)*l_(x))$\
      1. $Q #sym.arrow.l W_q X + b_q$\
      2. $K #sym.arrow.l W_k X + b_k$\
      3. $Q #sym.arrow.l W_v X + b_v$\
      4. $S #sym.arrow.l K^T Q$\

      *return* $V dot "softmax"(S/sqrt(d_"attn"))$
    ]
  ),
  caption: [
  Algorithm: Attention
  ]
)

== Autoregressive

#let box_words(text) = {
  let words = text.split(" ")
  for word in words {
    box(inset: 5pt, stroke: 0.1pt, radius: 0pt)[#word]
  }
}

#figure(
diagram(
spacing: (10mm, 5mm), // wide columns, narrow rows
edge-stroke: 0.5pt, // make lines thicker
node-corner-radius: 3pt,
mark-scale: 60%, // make arrowheads smaller
  node((0, 0), [#box_words("Sun rises from")]),
  edge((0, 0), "r", "-|>", [input] ),
  node((1, 0), [LLM], stroke: 1pt, fill: base ),
  edge((1, 0), "r", "-|>" ),
  node((2, 0), [#box_words("the")]),

  edge((2,0), (2, 1), (0.35, 1),(0.35, 2), "-|>" ),

  node((0, 2), [#box_words("Sun rises from the")]),
  edge((0, 2), "r", "-|>", [input] ),
  node((1, 2), [LLM], stroke: 1pt, fill: base),
  edge((1, 2), "r", "-|>" ),
  node((2, 2), [#box_words("east")]),

  edge((2,2), (2, 3), (0.45, 3), (0.45, 4), "-|>" ),

  node((0, 4), [#box_words("Sun rises from the east")]),
  edge((0, 4), "r", "-|>", [input] ),
  node((1, 4), [LLM], stroke: 1pt, fill: base),
  edge((1, 4), "r", "-|>" ),
  node((2, 4), [#box_words("<eol>")]),

  
),
caption: [
Example for an autogressive LLM policy. The LLM is trained to predict the
next most probably word. Then this is appended to the original string and
the process continues until the end of line special token is received.
]
)


For an input string ${s_(1)..s_(n)}$ and output string ${s_(n+1)..s_(n+k+1)}$,
a model works on the policy of:
$
  p(s_(n+1)..s_(n+k+1)|s_(1)..s_(n))\
  p(s_(n+1)| s_(1)..s_(n)) p(s_(n+2)| s_(1)..s_(n)+1) .. p(s_(n+k)| s_(1)..s_(n+k-1))
$ 

Thus each successive string is sampled based all the previous set of
strings in sequence. This policy can be generalized to simply predict the
next word in the string using the chain rule and then repeated. 

== Context Length
The total length of the input tokens is called the context length.
The context length can be considered as the short term memory of the llm
and the conditional tokens against which the llm predicts the next most
likely word.

Therefore a high context length is topic of much discussion.
Theoretically the length of input string can be infinite, however
in practice its limited due to several factors. For example LLama 3.3
@dubey2024llama family of models have an impressive context length of
$128 k$ tokens.


However the effective context length is often much shorter. 
// #cblock()[TODO]
// Find the effective context length.

=== Size of training dataset
  The documents in training dataset have a finite length. Most documents in
  popular datasets often do not exceed 10k tokens. LLM(s) trained on
  this context length often do not generalize over longer contexts.
  Increasing the
  context length above this during inference causes the performance
  of the model to degrade rapidly. @an2024does @ulvcar2021training

#show table.cell.where(y: 0): strong
#set table(
  stroke: (x, y) => if y == 0 {
    (bottom: 0.7pt + black)
  },
  align: (x, y) => (
    if x > 0 { center }
    else { left }
  )
)

#figure(
  grid(
    columns: (auto, auto),
    gutter: 1cm,
    table(
      columns: 3,
      table.header([Dataset Name],[Average Length],[Data Access]),

      [SumSurvey],  [>12,000 tokens], [#link("https://github.com/Oswald1997/SumSurvey")[SumSurvey Dataset]],
      [NarrativeXL],  [>50,000 words], [#link("https://arxiv.org/abs/2305.13877")[NarrativeXL Dataset]],
      [LongBench],  [~10,000 tokens], [#link("https://arxiv.org/abs/2308.14508")[LongBench Dataset]],
      [HiPool],  [4,034 tokens], [#link("https://arxiv.org/abs/2305.03319")[HiPool\ Benchmark]],
      [Databricks\ DocsQA],  [2,856 tokens], [#link("https://www.databricks.com/blog/long-context-rag-performance-llms")[Databricks Blog]]
    ),
    align(bottom)[
      #image("./titan.png"),
    ]
  ),
  caption: [
  Left: Average sizes of documents in few popular LLM training datasets\
  Right: Accuracy of popular LLM with increasing context length. Signifying effective context length. @behrouz2024titans
  ]
)

  

=== Quadratic complexity
  Transformer based attention architecture requires quadratic time complexity
  over the context length. Specifically $Q K^T$ multiplication requires $O(n^2)$
  computation and memory. This is the vanilla attention and here the output
  token can attend to all the input tokens. This can be visualized using the
  attention mask.@fournier2023practical

#figure(
  grid(
    columns: (auto, auto),
    rows: (auto, auto),
    gutter: 5pt,
    [],
    [#align(center)[Input Tokens]],
    [#align(center + horizon)[Output\ Tokens]],
    grid(
      fill: (x, y) => rgb(
        if (x == y) {
          "#7287fd"
        } else {
          "#eff1f5"
        }
      ),
      stroke: rgb("#4c4f69"),

      columns: (0.8em,) * 16,
      rows: 0.8em,
      align: center + horizon,

      ..([], [], [], [], [], [], [], [])
        .map(grid.cell.with(y: 15)),
    )
  ),
  caption: [Attention Mask of Vanilla Transformer]
)



== Representation of Data
Let the vocabulary of Model be: $V$. The input text/data needs to be represented
in this vocabulury on which the model would be trained. Empirical experiments
have shown that correct scaling of vocabulury size in relation to parameter
count of the model is very important for optimal performance. @takase2024large

The process of representing input data as a sequence of vocabulury elements
is called *tokenization*. Tokenization can be achieved in several manner.
For a piece of text: "Sun rises from the east", it can be tokenized as:
1. *Character-level Tokenization*:
  $
    V #sym.in [a,.., z]
  $

  Here each vocabulary token corresponds to a alphabet: ['S', 'u', ..'s', 't'].
  These generalize well, however these are too small representation and very expensive for large context
  lengths due to quadratic complexity. In addition a big vocabulary causes the
  embedding matrix to huge causing increasing memory complexity.

2. *Word-level Tokenization*:
  $
    V #sym.in "Words"
  $
  Here each vocabulary token corresponds to a word: ['Sun', 'rises', 'from', 'the', 'east'].
  These require very large vocabulury and do not generalize well.

3. *Subword Tokenization*:
  $
    V #sym.in "Commonly Occurring Sub Words"
  $
  Here each vocabulary token corresponds to a commonly occurring word segments: ['ris', ses', 'the' ..] 
  These offer the best middle ground between vocabulary size and generalization. Notably
  common words like _'the'_ are tokenized as it is.
4. *Byte-Pair Encoding (BPE)*:
  This is the most popular type of sub-word tokenization scheme. It was introduced
  in the paper: Neural Machine Translation of Rare Words with Subword Units @sennrich2015neural

= Diffusion



Diffusion models are based on non-equilibrium thermodynamics. Here we iteratively
add noise to the original data (forward diffusion process) and then learn to
iteratively get the data back from noise (backward diffusion process).

#figure(
  image("./diffusion.png"),
  caption:
    [
      Noise is progressively added to an image and then a model
      is trained to progressively remove noise. @yang2023diffusion
    ]
)

== Forward Diffusion Process
For data distribution: $x_0$ in $q(x)$ during diffusion process we add
Gaussian noise to sample in $T$ steps, producing a sequence of noisy samples
$x_1, .., x_T$.
$
q(x_t | x_{t-1}) &= N(x_t; sqrt(1 - \beta_t) x_{t-1}, \beta_t I)\
q(x_{1:T} | x_0) &= sum_(t=1)^T q(x_t | x_(t-1))
$

Here the step size is controlled using variance schedule: ${beta_t in (0, 1)}_(t=1)^T$
hyperparameters. Key to the success of this sampling process is training the reverse Markov chain to match the actual time reversal
of the forward Markov chain.

#figure(
  diagram(
  spacing: (10mm, 5mm), // wide columns, narrow rows
  node-stroke: 1pt, // outline node shapes
  edge-stroke: 1pt, // make lines thicker
  node-corner-radius: 10pt,
  mark-scale: 60%, // make arrowheads smaller

  node((2, 0), $X_T$),
  node((3, 0), "...", stroke: 0pt),
  edge((2,0), (3,0), "->"),
  node((4, 0), $X_t$),
  edge((3,0), (4,0), "->"),
  edge((4,0), (6,0), "->", $p_theta(X_(t-1)|x_t)$),
  edge((6,0), (4,0), "-->", bend: 35deg, $q(x_t|x_(t-1))$),
  edge((4,0), (6,0), "-->", bend: -85deg, stroke: red, $q(x_t|x_(t-1)) "is unknown"$),
  node((6, 0), $X_(t-1)$),
  edge((6,0), (7,0), "->"),
  node((7, 0), "...", stroke: 0pt),
  edge((7,0), (8,0), "->"),
  node((8, 0), $X_0$, fill: gray),

  node(enclose: ((4,0), (6,0), (6, 4)), stroke: (thickness: 0.1pt, dash: "dashed"), inset: 10pt,snap: false), // prevent edges snapping to this node
  ),
  caption: [Forward Diffusion Process @weng2021diffusion]
)

== Backward Diffusion Process

If we can reverse the above process and sample from $q(x_{t-1} | x_t)$,
we will be able to recreate the true sample from a Gaussian noise input,
$x_T approx N(0, I)$. Note that if
$beta_t$ is small enough, $q(x_(t-1) | x_t)$ will also be
Gaussian. Unfortunately, we cannot easily estimate $q(x_(t-1) | x_t)$
because it needs to use the entire dataset and therefore we need to learn a model
$p_theta$ to approximate these conditional probabilities in order to run the
reverse diffusion process.

$
  p_theta(x_(0:T)) = p(x_T) sum_(t=1)^(T) p_theta(x_(t-1) | x_t) p_theta(x_(t-1) | x_t) = N(x_(t-1); u_theta(x_t, t), sum_theta(x_t, t))
$

== Parameterization of $L_t$ for Training Loss

Now for training a neural network which can learn to create an image from
noise, we need to define a loss function, such that a model can learn to
minimize it. The mathematical loss function is very complex, however
empirically found simplified loss functions are used. @ho2020denoising

$
L_t^("simple") &= E_(t dash.wave [1, T], x_0, epsilon_t) [ |epsilon_t - epsilon_theta(x_t, t)\|^2 ]\
&= E_(t dash.wave [1, T], x_0, epsilon_t) [ \|epsilon_t - epsilon_theta(sqrt(alpha_t) x_0 + sqrt(1 - alpha_t) epsilon_t, t)\|^2]\
$

Thus final objective becomes:
$
  L_("simple") = L_t^("simple") + C
$


#figure(
  grid(
    columns: (auto, auto),
    gutter: 1cm,
    table(
      columns: 3,
      table.header([*Algorithm:* Diffusion Training]),

      [
      *Require*: Model $x_(theta)(z_t)$ to be trained\
      *Require*: Data set $D$\
      *Require*: Loss weight function $w()$\

      *while* not converged *do*
        Sample Data : $x approx D$\
        Sample Time : $t approx U[0, 1]$\
        Sample Noise: $e approx N(0, I)$
        Add Noise   : $z_t = alpha_t x + sigma_t e$\
      
        Clean data is target for $x_t = x$\
        Log-SNR: $lambda_t = log[alpha_t^2 / sigma_t^2]$ \
        Loss: $L_theta = w(lambda_t) \|tilde{x} - x_theta(z_t)\|_2^2$\
        Optimization: $theta #sym.arrow.l theta - gamma nabla_theta L_theta$\
      *end while*
      ]
    ),
    
    table(
      columns: 3,
      table.header([*Algorithm:* Sampling]),

      [
        1: $x_T dash.wave N(0, I)$\
        2: for $t = T, ..., 1$ do\
        3: $z dash.wave N(0, I)$ if $t > 1$, else $z = 0$\
        4: $x_(t-1) = (1)/(sqrt(alpha_t)) ( x_t - (1 - alpha_t)/(sqrt(1 - alpha_t)) epsilon_theta(x_t, t)) + sigma_t z$\
        5: end for\
        6: return $x_0$
      ]
    ),
  ),

  caption: [
    Algorithm: Training and Sampling Diffusion Models @ho2020denoising
  ]
)

= Text Diffusion: Future Outlook
For text generation auto-regressive models are most widely used. However
they can not be parallelized. This if fine for text generation which is
fundamentally discrete in nature. However for data with high dimension
and resolution such as images, auto-regressive paradigmn is very slow.
Thus they use diffusion based generative models to make effective use of
parallelism during training and inference.

However, with the rapid growth in the user of LLM and test time scaling @muennighoff2025s1
there ability to parallelise text generation is major consideration. Diffusion
process is continuous in nature. For modelling text based discrete diffusion
model, technique of score entropy is used.

== Discrete Diffusion Modelling Techniques

*Mean Prediction*: Instead of directly parameterizing the
ratios $p_t(y)/p_t(x)$ ,@austin2021structured 
instead follow a strategy of to learn the reverse density $p_0|t$. This actually recovers the ratios
$p_t(y)/p_t(x)$ in a
roundabout way, but comes
with several drawbacks. Learning $p_0|t$ is inherently
harder since it is a density (as opposed to a general value).
Furthermore, the objective breaks down in continuous time
and must be approximated @campbell2022continuous. As a
result, this framework largely underperforms empirically.

*Ratio Matching*: Originally introduced in @hyvarinen2005estimation
, ratio matching learns
the marginal probabilities of each dimension with maximum
likelihood training. However, the resulting setup departs
from standard score matching and requires specialized and
expensive network architectures.
As such, this tends to perform worse than mean prediction.

*Concrete Score Matching*: @meng2022concrete generalizes
the standard Fisher divergence in score matching, learning
with concrete score matching:
Unfortunately, the loss is incompatible with the fact that
$p_t(y)/p_t(x)$ must be positive. In particular, this does not sufficiently
 penalize negative or zero values, leading to divergent
behavior. Although theoretically promising, Concrete Score
Matching struggles.

== Score Entropy

Build on top of previous methods, score entropy aims to mitigate the problems
faced in previous techniques. It is the current state of art method for
discrete diffusion modelling, originally introduced in @lou2023discrete.

It has several favourable properties which can be used to derive
loss function for discrete diffusion steps.

*Properties*:
  1. *First*, score entropy is a suitable loss function that recovers
    the ground truth concrete score.
  2. *Second*, score entropy directly improves upon concrete
    score matching by rescaling problematic gradients.
  3. *Third*, similar to concrete score matching, score entropy
    can be made computationally tractable by removing the
    unknown $p(y)/p(x)$ term. There are two alternative forms, the
    first of which is analogous to the implicit score matching
    loss. @hyvarinen2005estimation
  4. *Fourth*, the score entropy can be used to define an ELBO
    for likelihood-based training and evaluation.
  5. *Fifth*, score entropy can be scaled to high dimensional tasks.

Using these the denoising loss function becomes:
$
E_(x_0 \sim p_0 \ t \sim U(0, 1] \ x dash.wave p_t(dot | x_0)) [ sum_(x eq y) w_(x y) ( s_theta(x)_y - (p(y | x_0))/(p(x | x_0)) log s_theta(x)_y ) ]
$

== Comparing against Auto-regressive Model

1. *Unconditional generation quality*\
  We compare SEDD and GPT-2 based on their zeroshot perplexity on test datasets and based on the
  likelihood of generated text using a larger model
  (GPT-2 large, 770M parameters), similar to Savinov et al. (2022); Strudel et al. (2022); Dieleman
  et al. (2022). As observed by Lou et al. (2023),
  SEDD produces text with lower perplexity than
  GPT-2 without annealing. When sampling with
  1024 steps, SEDD produces text of similar likelihood as annealed sampling from GPT-2. See tables 1 and 3 for more details.
2. *Conditional generation quality*\
  We evaluate the conditional generation quality using automated metrics and the lm-eval-harness
  suite of Gao et al. (2021).
  \
  *Automated metrics* We compare the quality of
  text generated by SEDD and GPT-2 using the
  MAUVE metric (Pillutla et al., 2021) and the perplexity of sampled continuations using GPT-2 large.
  We compute those metrics using prefixes and continuations of 50 tokens each. We extract 1000
  prompts from the test set of OpenAI’s webtext
  dataset. While Lou et al. (2023) suggests that
  SEDD medium matches GPT-2 in quality, we observed that SEDD achieves slightly lower MAUVE
  score. Reproducing the MAUVE results of Lou
  et al. (2023) was non-trivial, and we observed that
  generating more tokens achieves better quality than generating 50 only, even if we use 100
  tokens to compute MAUVE. Indeed, sampling
  the 50 tokens of the continuation only resulted in
  poorer performance. For more details on this surprising result and exact MAUVE values, refer to
  appendix A.4.
  \
  *Downstream performance* We compare the accuracy of SEDD and GPT-2 on downstream tasks
  using the evaluation suite of Gao et al. (2021). The
  results are shown in table 2. We noticed the performance of SEDD and GPT-2 overall are very close1.
  \
  *Diversity* We evaluate diversity based on statistics computed on the training data and synthesized
  samples. See table 4 for all details. Surprisingly,
  the repetition rate of SEDD increases as the number
  of sampling steps increases. Similarly, the unigram
  entropy negatively correlates with the number of
  sampling steps. Importantly, SEDD appears less
  diverse than GPT-2 with and without annealing.
3. *Sampling speed*\
  We compare the generation latency of SEDD and
  GPT-2 with KV-caching on a single NVIDIA A100SXM4-40GB GPU in fig. 1. See appendix A.3 for
  more details

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: horizon,
    inset: 5pt,
    stroke: stroke(0.7pt),
    [Model], [params], [LAMBADA], [Wikitext2], [PTB], [WikiText103], [1BW],
    [GPT-2 (small)], [124M], [#strong[44.663]], [39.576], [129.977], [39.576], [#strong[52.808]],
    [SEDD (small)], [170M], [≤ 51.146], [#strong[38.941]], [#strong[110.97]], [#strong[38.887]], [≤ 78.67],
    [GPT-2 (medium)], [255M], [#strong[35.879]], [29.575], [116.516], [29.575], [#strong[43.627]],
    [SEDD (medium)], [424M], [≤ 42.385], [#strong[28.855]], [#strong[83.411]], [#strong[28.54]], [≤ 60.97],
    [GPT-2 (large)], [774M], [33.969], [26.664], [139.427], [26.664], [44.783],
  ),
  caption: [Zero-shot test perplexity]
)

#let colors = (
  rgb("#002833"),
  rgb("#002833"),
  rgb("#d95f0e"),
  rgb("#2ca02c"),
  rgb("#1f78b4"),
  rgb("#6a3d9a"),
  rgb("#b15928"),
  rgb("#e31a1c"),
  rgb("#ff7f00"),
  rgb("#a6cee3"),
  rgb("#fb9a99"),
  rgb("#cab2d6"),
)

#let data = (
  ([Mercury Coder Mini], 1109),
  ([Mercury Coder Small], 737),
  ([Owen2.5 Coder 7B], 207),
  ([Gemini 2.0 Flash-Lite (Preview)], 201),
  ([Codestral (Jan 25)], 171),
  ([Llama 3, 1 8B], 153),
  ([Nova Micro], 148),
  ([Mistral Small V2 Lite], 126),
  ([DeepSeek Coder], 93),
  ([Claude 3.5 Haiku], 61),
  ([GPT-4o mini], 59),
  ([Command-R+], 54),
)

== Industry Applications

#figure(
canvas({
draw.set-style(
    axes: (grid: (stroke: 0.5pt + luma(200), dash: "dashed")),
  )
  chart.barchart(
    mode: "basic",
    size: (8, 10),
    label-key: 0,    // Now gets the string label
    value-key: 1,
    data,
    bar-style: idx => ( // Function: index -> style dictionary
      fill: colors.at(idx), // Set fill color
      stroke: black, // Add outline to each bar
    ),
    horizontal: true,
    x-tick-set: 150,
    x-grid: true,

  )
}),
caption: [*Speed Comparison*: Output Tokens per Second\
*Mercury Coder* which are first large scale discrete diffusion models are
significantly faster than auto-regressive models.

]
)


A significant area of current development, discussed in Section 5, is the application of diffusion principles to discrete data such as text and code. While autoregressive models are established here, their sequential generation can be slow. Discrete diffusion techniques, including the promising Score Entropy method (Section 5.2), aim to overcome this limitation. As illustrated in Figure 5, recent discrete diffusion models like Mercury Coder demonstrate substantial improvements in generation speed (output tokens per second) for tasks like code generation compared to several autoregressive counterparts. This progress signifies the potential for diffusion-based approaches to offer faster and potentially more parallelizable alternatives in domains traditionally dominated by autoregressive generation, impacting future tools for software development, writing assistance, and more.

= Conclusion
This review has examined two cornerstone generative techniques in modern deep learning: autoregressive models, largely enabled by the Transformer architecture, and diffusion models. We explored the foundational components of the vanilla Transformer, the principles of autoregressive sequence generation common in LLMs, and associated considerations like context length and data representation. Subsequently, the review detailed the iterative noising (forward) and learned denoising (backward) processes underpinning diffusion models, including loss parameterization.

In conclusion, both autoregressive and diffusion paradigms represent powerful approaches to generative modeling, each with distinct strengths and areas of active research. The successful application of diffusion to discrete data marks a significant development, opening new possibilities for efficient and high-quality generation beyond continuous domains. As these techniques continue to evolve and potentially converge, they will undoubtedly fuel further breakthroughs across a wide spectrum of AI-driven applications.

#bibliography("works.bib", full: true)
