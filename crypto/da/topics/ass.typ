#import "lib.typ": *
#import "@preview/fletcher:0.5.5": *


#let base = rgb("#eff1f5")

#show: doc => assignment(
  title: "Review of Generative Techniques in Deep Learning",

  course: "Cryptography and Network Security Lab",

  // Se apenas um author colocar , no final para indicar que é um array
  author: "Apurva Mishra: 22BCE2791",
  date: "3 March, 2025",
  doc,
)

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
    Algorithm: Represetnting input tokens into latent space
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
    Algorithm: Converting vector from latent space to vocablury
    ]
  )

*Layer Normalization*
For all: encoder, sub-layers and decoder layer normalization is performed for
the final output. This is essential as transformer have shown to become
unstable due to changes in activations from variying dataset. @berlyand2021stability
Here layer normalization ensure that each output is normalized to a consistent
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
      *Output*: $V #sym.in R^(d_("out")*l_(x)),$, updatead represetations of tokens in X, folding in information from token *Z*\ 

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
strings in sequence. This policy can be generalizsed to simply predict the
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


However the effective context length is often much shorter at just 
#cblock()[TODO]
Find the effective context length.

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
Let the vocablury of Model be: $V$. The input text/data needs to be represented
in this vocablury on which the model would be trained. Empirical experiments
have shown that correct scaling of vocablury size in relation to parameter
count of the model is very important for optimal performance. @takase2024large

The process of representing input data as a sequence of vocablury elements
is called *tokenization*. Tokenization can be achieved in several manner.
For a piece of text: "Sun rises from the east", it can be tokenized as:
1. *Character-level Tokenization*:
  $
    V #sym.in [a,.., z]
  $

  Here each vocablury token corresponds to a alphabet: ['S', 'u', ..'s', 't'].
  These generalize well, however these are too small representation and very expensive for large context
  lengths due to quadratic complexity. In addition a big vocablury causes the
  embedding matrix to huge causing increasing memory complexity.

2. *Word-level Tokenization*:
  $
    V #sym.in "Words"
  $
  Here each vocablury token corresponds to a word: ['Sun', 'rises', 'from', 'the', 'east'].
  These require very large vocablury and do not generalize well.

3. *Subword Tokenization*:
  $
    V #sym.in "Commonly Occurring Sub Words"
  $
  Here each vocablury token corresponds to a commonly occurring word segments: ['ris', ses', 'the' ..] 
  These offer the best middle ground between vocablury size and generalization. Notably
  common words like _'the'_ are tokenized as it is.
4. *Byte-Pair Encoding (BPE)*:
  This is the most popular type of sub-word tokenization scheme. It was introduced
  in the paper: Neural Machine Translation of Rare Words with Subword Units @sennrich2015neural

= Diffusion
Diffusion models are based on non-equilibrium thermodynamics. Here we iteratively
add noise to the original data (forward diffusion process) and then learn to
iteratively get the data back from noise (backward diffusion process).

== Forward Diffusion Process

== Backward Diffusion Process
#bibliography("works.bib", full: true)
