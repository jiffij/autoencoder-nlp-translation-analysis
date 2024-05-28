# autoencoder-nlp-translation-analysis

## Overview of NLP Classification Task and Models
In this experiment, we will discover different approach
of seq2seq model for machine translation.

## Dataset Preparation
The EnglishFrench dataset is collected form Tatoeba,
consisting of 232736 English and French sentence
pairs. Since we are comparing the performance of
models, we do not need much data to improve performance.
Therefore, only sentences starting with
thr form ”(pronoun) (be)” and less than 15 words
are used. The trimmed size is 22907. with 7K and
4.6K French and English vocabulary, which are further
converted to numerical representations. The
train:test ratio is 9:1.

## Result
According to request, different seq2seq model are
compared and evaluated based on their performance
on French to English translation. Table 1 are the
train and test result for the different models. Beside,
The training loss and sample output of each model
are recorded and placed in Appendix A for further
analysis.
Table 1 compares the performance of different
seq2seq models on a test dataset using various metrics
such as F-measure, precision, and recall for both
rogue 1 and rogue 2. The original seq2seq model
achieves a moderate R1 F-measure of 61.9%, while
the LSTM variant shows improvement with an R1
F-measure of 65.76%. The bi-LSTM model further
improves to an R1 F-measure of 66.7%. However,
the addition of attention mechanisms in the seq2seq
model leads to mixed results, and the Transformer
encoder variant performs the poorest with an R1 Fmeasure
of 51.03%. Overall, the table highlights the
impact of different model architectures on the performance
of seq2seq models, with the bi-LSTM model
showing the best performance among the variants.

## Analysis
In the following, we will compare different models
and methods and analyze the root cause of their
varying performance on the machine translation task.
We examine the performance of different attention
mechanisms in seq2seq model: including using GRU,
LSTM, BI-LSTM as encoder and decoder, GRU with
attention decoder and transformer encoder. The
method will be compared according to Rogue evaluated
metric, the training loss and direct observation.
In order to maintain a controlled environment
for comparison, all experiments were conducted using
the same hyperparameter, loss function and optimizer,
if not explicitly stated. The default setting
1
| Model                       | R1 F   | R1 P   | R1 R   | R2 F   | R2 P   | R2 R   |
|-----------------------------|--------|--------|--------|--------|--------|--------|
| seq2seq (original)          | 84.83  | 78.36  | 92.87  | 76.72  | 69.62  | 86.07  |
| seq2seq (LSTM)              | 84.18  | 78.21  | 91.57  | 76.09  | 69.49  | 84.75  |
| seq2seq (bi-LSTM)           | 85.58  | 79.35  | 93.29  | 78.13  | 71.2   | 87.2   |
| seq2seq (Attention)         | 88.18  | 81.48  | 96.35  | 83.14  | 75.54  | 92.92  |
| seq2seq (Transformer Encoder)| 52.69  | 51.4   | 56.19  | 34.85  | 33.38  | 38.39  |
| seq2seq (BI-LSTM + Attention)| 84.77  | 78.55  | 92.38  | 77.18  | 70.34  | 85.99  |
| seq2seq (original)          | 67.21  | 62.51  | 73.48  | 50.36  | 46.0   | 56.47  |
| seq2seq (LSTM)              | 66.82  | 62.75  | 72.24  | 50.14  | 46.3   | 55.53  |
| seq2seq (bi-LSTM)           | 68.1   | 63.65  | 73.95  | 51.68  | 47.44  | 57.53  |
| seq2seq (Attention)         | 69.33  | 64.65  | 75.46  | 53.79  | 49.3   | 59.92  |
| seq2seq (Transformer Encoder)| 51.03  | 50.1   | 53.16  | 30.84  | 29.82  | 32.96  |
| seq2seq (BI-LSTM + Attention)| 67.88  | 63.39  | 73.74  | 51.39  | 47.18  | 57.12  |
Table 1: Test Results for Different Models. F: f measure, P: precision, R: recall, R1: rogue 1, R2: rogue 2

| Parameter          | Value       |
|--------------------|-------------|
| Loss function      | NLLLoss     |
| Optimizer          | SGD         |
| No. of Epoch       | 10          |
| Early stopping     | 2           |
| Learning rate      | 0.01        |
Table 2: Default Settings
is as follow.

### A little history
Before emergence of seq2seq model, Statistical Machine
Translation was the common way to the task.
Hidden Markov Model is one very common SMT
model, which uses Maximimum Likelihood Estimation
to estimate the probability of the source
word translating into the target. And also the cooccurance
of lexicons with other words.
However, these statistical model have limited understanding
on the context of the sentence and often
sound influent. The short-range dependencies and
lack of deep understanding is the root of the bad
translation.

### seq2seq (original)
seq2seq model make use of a encoder-decoder structure
to allow the encoder capture the full context
semantic of the whole sentences, which help to distinguish
the lexicons in a sentence. Also it make use of
RNN architecture to enhance the dependency range
and is flexible to any length of input.
The original GRU seq2seq model has an overall
performance, with F1 and F2 score of 67.21% and
50.36%. According to the Appendix A, the mode
is able to translate simple sentences like ”vous etes
tires d affaire .” (you re off the hook .). It could also
preserve certain meanings and SVO sentence structure.
However, when it comes to longer sentences,
repetition is observed, with words like ”ve” being incorrectly
duplicated as ”ve ve” in the translations.
Additionally, in the sentence ”i m grateful for everything
you ve done for me .”, the model translate last
part as ”i ve ve done for you .”. In which the GRU encoder
has captured the incorrect meaning. And also
failed in capturing per word semantic, ”economics”
as ”science”, which are two distinct concept. Finally,
there are instances of incomplete phrases like ”we re
not lost . i know where we are .” translated as ” we
re not only to . . .”, which the second sentence is
missing, further highlighting the model’s limitations
in understanding the full semantic of long sentences.

### seq2seq (LSTM)
Of course, some may wonder if the problem of GRU
seq2seq model is due to lack of training samples. In
2
order to verify the thought, we replaced the GRU
modules in encoder and decoder with LSTM. According
to Table 1, the training and validation F scores
are similar to GRU seq2seq.
According to Figure 3 in Appendix B, the training
curve of the LSTM seq2seq shows a exponential
decay. Appendix A shows the example translation,
the LSTM model generally produces mostly accurate
translations. Most examples from the LSTM model
accurately convey the meaning of the original French
sentences. Unlike the original model, the LSTM
model doesn’t exhibit repetition issues in these examples.
Despite there are different outcome to the
target, like ”we re just students .” translated as ”we
re only students .”, the meaning is correct, where
”just” and ”only” have similar meaning. Showing a
high level of under standing on word and sentence
level. Missing word issue still exist.
It is not surprising that LSTM performs better
than GRU. Since GRU replaced the forget and input
gate with a single update gate, which reduces
the model size and computational cost. But memorization
mechanisms might be less sophisticated and
having one information high-way (cell state) instead
of two (cell and hidden state). Resulting possibly
more information loss and shorter dependency range.

### seq2seq (Bi-LSTM)
Next we replace the GRU in encoder of the original
seq2seq model with a bidirectional LSTM module.
It has higher training and testing F, precision,
and recall than both seq2seq (original) and (LSTM)
by 1 to 2 percent. Appendix A shows the output
of seq2seq BI-LSTM. The result is already quite impressive
for BI-LSTM. Most of the randomly sampled
testing sentences are correct, with only an incorrect
word. Translating ”sourire” as ”dying”, that should
mean ”smiling”, implies that BI-LSTM still struggles
a bit in understanding word level meanings. Besides,
seq2seq (BI-LSTM) success in capture understanding
sentence context and grammatical feature.
The improvement could be likely attributed to the
bidirectional nature of the model, where the encoder
no longer assumes causal dependencies of the input.
This would enlarge the dependency range of the encoder,
and become able to relate not only preceding,
but following words. Improving the ability of capturing
full sentence semantic and word relation. This
explains the phenomenon of high accuracy on complex
sentences.

### seq2seq (Attention)
After modifying the encoder with BI-LSTM, we modify
the decoder in the original seq2seq to perform attention
mechanism.
(forward flow)
Figure 1: Training loss of seq2seq (Attention) without
Cosine Annealing (left), with Cosine Annealing
(right)
In the first run with the default setting in Table
2, the training loss spikes 150K iteration as shown
in Figure 5(left), which might cause by the large
constant learning rate when the model is close to
a minimum, a large learning rate will lead to escap
from minimum. To ensure a smoother descent,
we applied a cosine annealing scheduler with linear
warmup. The a smooth loss curve is observed as Figure
5b.
The seq2seq (Attention) demonstrated highest
train and validation Rogue scores compare to previous
models as shown in Table 1. And demonstrated
nearly perfect translation according to Appendix A.
The only difference appears in the translation ”we re
baffled .”, translated as ”we re confused . ”. In this
case, ”baffled” and ”confused” have the same meaning.
Showing the model had correctly capture all the
semantic of word and sentence, and also the grammatical
relation.
The outstanding result might correspond to the attention
mechanism. In which attention ignored the
3
sequential distance, rather, it sees all word at once
with global attention. This approach would provide
infinite dependency range, and allow more dynamic
correlation between each input position. Resulting
better learned word correlation, like relating ”perplexes”
to ”baffled” and ”confused”. This is why
attention improves model performance on complex
sentence structure.

### seq2seq (Transformer)
Furthermore, we have also change the encoder in the
original seq2seq to a transformer encoder. The flow
is rather simple, the sinusoidal positional encoding
is added to the word embedding and pass through a
transformer encoder. Finally, average the output in
zero dimension.
Surprisingly, the transformer does not perform as
well as other model. It has the lowest Rouge score
according to Table 1. Also, the translation of seq2seq
(Transformer) is rather inaccurate. It could not completely
understand the word and sentence meaning.
For instance, translating ”allowed to park here” as
”allowed in the park”. The target means the person
is parking a car, while the translation implying a
person being inside the park area.
According Figure 6 in Appendix B, the fluctuating
training loss implies that the model struggle to
converge. The instability might possibly ascribable
to the averaging bottleneck between the encoder and
the decoder, combining representation through averaging
might change the meaning of the embedding.

### Improvement: seq2seq (BILSTM + Attention)
In light of the good performance of the BI-LSTM and
Attention. We would like to experiment a simple improvement
by simply combining these BI-LSTM encoder
and Attention decoder. According to Table
1, the resulting model does not seems to improve,
with F1 and F2 scores of 67.88 and 51.39. The performance
is similar to simply using a seq2seq (BILSTM).

### Summary
This experiment compared different seq2seq models
for French to English translation. The best performing
model used a Bi-LSTM encoder, achieving good
understanding of sentence structure and meaning.
Adding attention to the decoder further improved
results, suggesting attention helps capture word relationships
in complex sentences. Surprisingly, a
Transformer encoder underperformed, possibly due
to convergence issues or the way it combines encoded
representations. Combining Bi-LSTM and attention
didn’t significantly improve performance over the Bi-
LSTM model alone. Overall, this study highlights
the effectiveness of Bi-LSTM encoders and attention
mechanisms for seq2seq machine translation.