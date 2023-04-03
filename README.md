# experiments-coreml-gpt2

Following on from https://github.com/anentropic/experiments-coreml-ane-distilbert I wanted to experiment with a more interesting model.

## Chapter 1
<details>
<summary>Experimenting with <code>swift-coreml-transformers</code> (fail...!)</summary>

https://github.com/huggingface/swift-coreml-transformers

The CoreML versions of these models aren't uploaded to Huggingface Hub, so we need to manually fetch them from the GitHub repo above.

`gpt2-512.mlmodel` has 341M parameters ([maybe?](https://github.com/huggingface/swift-coreml-transformers/issues/19#issuecomment-1493061055)) making it roughly equivalent to GPT-2-Medium (355M params).

Two other sizes are provided: `gpt2-256.mlmodel` (similar to base GPT2 "small") and `gpt2-64-12.mlmodel` (smaller than any OpenAI GPT2 model).

Even the smallest, `gpt2-64-12.mlmodel` takes 3 minutes to load (!) Now I appreciate why everyone uses notebooks...

We soon run into problems - since the .mlmodel is derived from a from-scratch rewrite of GPT2 in Swift, there seem to be some differences.

- the HF tokenizer gives inputs with `input_ids` and `attention_mask` arrays, but the mlmodel expects `input_ids` and `position_ids`. I guessed the latter are equivalent, but not sure.
- it looks like the tokenizer vocab is different between the [Swift-CoreML](https://github.com/huggingface/swift-coreml-transformers/blob/079477b014f3a416914888d829460c1a571556b3/Resources/gpt2-vocab.json) implementation and the [usual one](https://huggingface.co/openai-gpt/raw/main/tokenizer.json).  I think this explains the nonsense I got when trying to decode the `output_logits`.
- Because it's in Swift I can't easily use their tokenizer for this, I want to use a Huggingface one

It seems like my best bet is to ignore this repo (which hasn't been touched since 2019, and so will not have any ANE-friendly optimisations anyway) and instead use `coremltools` to convert an original OpenAI GPT2 model to coreml format.
</details>

## Chapter 2
<details>
<summary>1. Experimenting with HuggingFace <code>openai-gpt</code> (TLDR; this is the wrong model)</summary>

I first tried the model at https://huggingface.co/openai-gpt

Just experimenting in ipython console at first, I tried the "how to use this model in PyTorch" example code:

```python
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import torch

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
model = OpenAIGPTModel.from_pretrained("openai-gpt")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
```

Firstly, this loads pleasingly fast, especially compared to the 3 minute nightmare of the swift-coreml-transformers.

It's not clear what `last_hidden_states` represents, but I assumed maybe it contains token ids that can be decoded back to string via the tokenizer. I did not quickly find any instructions via Google, I asked ChatGPT but it started going round in circles giving me recipes that didn't work - usually a sign I'm on the wrong track.

Meanwhile Im aware I'm ignoring the more obvious code example above it:

```python
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='openai-gpt')
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
```

It works but gives low-quality completions, such as:

> Hello, I'm a language model,

> he said, when i was finished.'ah well,'said the man,'that's

The reason is likely found in this message that printed when we initialised the pipeline:

> Some weights of OpenAIGPTLMHeadModel were not initialized from the model checkpoint at openai-gpt and are newly initialized: ['position_ids']
> You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

I'd originally picked this model because it sounded more 'official'. Actually now I am thinking this is actually GPT-1 🤦🏻‍♂️🤦🏻‍♂️🤦🏻‍♂️
</details>

<details>
<summary>2. Experimenting with HuggingFace <code>gpt2</code> (👍)</summary>

### Generate something


Let's try the same with https://huggingface.co/gpt2 instead:

```python
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
```

> Hello, I'm a language model,

> I'm writing a new language for you. But first, I'd like to tell you about the language itself

Much better!

I believe this is the "small" 124M param version of the model, some related model names on HF are: `gpt2-medium`, `gpt2-large`, `gpt2-xl`.

It seems zippy enough that I'll probably try a larger one once I get things working.

There is also `distilgpt2` which [apparently](https://transformer.huggingface.co/model/distil-gpt2) _"weighs 37% less, and is twice as fast as its OpenAI counterpart, while keeping the same generative power"_. So that could be worth a try, though I think only "small" version exists.

### Towards converting to CoreML

So, I'm hoping to use the `coremltools` to convert the PyTorch GPT2 to a CoreML model, to compare how it runs.

It seems like I will need to look inside the pipeline.

I can see that the basic idea of decoding tokens was correct: https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_generation.py#L270

I don't find any mention of `last_hidden_states`, instead the model returns a key `generated_sequence`. I guess this is due to different way they call the model here: https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_generation.py#L251

From here we can work up a modified example:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
text = "Hello, I'm a language model,"
encoded_input = tokenizer(text, return_tensors='pt')

generated_sequence = model.generate(
    input_ids=encoded_input['input_ids'],
    attention_mask=encoded_input['attention_mask'],
    max_length=30,
)
out_b = generated_sequence.shape[0]
in_b = encoded_input['input_ids'].shape[0]
generated_sequence = generated_sequence.reshape(
    in_b, out_b // in_b, *generated_sequence.shape[1:]
)

generated_sequence = generated_sequence[0].numpy().tolist()

records = [
    tokenizer.decode(
        sequence,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    for sequence in generated_sequence
]
print(records)
```

This gives us:

```python
["Hello, I'm a language model, not a programming language. I'm a language model. I'm a language model. I'm a language model"]
```

So it "works", but the generation quality is worse for some reason.

(I also tried with `num_return_sequences=5` but got "`ValueError: num_return_sequences has to be 1, but is 5 when doing greedy search.`"... not sure what that means)

I asked ChatGPT for help and it suggested I need to do "top-p sampling" and gave a modified code:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "Hello, I'm a language model,"
encoded_input = tokenizer(text, return_tensors='pt')

# Set the sampling parameters
temperature = 1.0
top_k = 0
top_p = 0.9

# Generate text
output_sequences = model.generate(
    input_ids=encoded_input['input_ids'],
    attention_mask=encoded_input['attention_mask'],
    max_length=30,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    repetition_penalty=1.0,
    do_sample=True,
    num_return_sequences=5,
)

# Decode generated text
generated_sequences = []
for generated_sequence in output_sequences:
    generated_sequence = generated_sequence.tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    total_sequence = (
        text[len(tokenizer.decode(encoded_input['input_ids'][0], clean_up_tokenization_spaces=True)) :]
    )
    generated_sequences.append(total_sequence)

print(generated_sequences)
```

This actually worked perfectly (and also showed me how to return multiple answers from `generate`), it gives results like:

```
[' I\'m not speaking to your computer, I\'m speaking to your text."\n\nHe touched his yellow leather',
 ' so naturally I started to think about it and come up with the idea to write some models of a language called',
 " so once I learn python in this context I'm excited about python, the language I'm going to be using",
 " and I'm a saying unit. I'm drawing the game. It's a game in a very visual style",
 " not a language scientist. If the writing algorithms on the above logic weren't incorrect and the writers were really interested"]
 ```

I think these are not quite as good as the first output value from the pipeline still (_"I'm writing a new language for you. But first, I'd like to tell you about the language itself"_) but perhaps it's just a question of tweaking the parameters.

Some info about these parameters:
- https://docs.cohere.ai/docs/controlling-generation-with-top-k-top-p
- https://huggingface.co/blog/how-to-generate  
  this one also suggests some other `generate` params to try

Using `temperature=0.85`, `top_k=50` and `top-p=0.99` I can get output like:

```
[" one of the best ever. I've worked on writing a lot of languages. But I've also done a",
 " a language model of software. I think it's good for me, because I get to have this little little",
 " so I'm interested in the various types of languages and how they interact with each other. I think it's",
 ' not a science fiction or fantasy, so when you write a character and you want to change their brain function,',
 " I'm a framework for building applications for web servers. I wrote this in Java at the beginning, but now"]
```

These are looking good now!

### Will it be possible?

In the meantime I have seen there may be some problems converting GPT-2 to CoreML: https://discuss.huggingface.co/t/conversion-to-coreml-for-on-device-use/13284

I also found this: https://github.com/huggingface/exporters/

HF made a library for converting their `transformers` models to CoreML 👍🎉  It's about 1 year old at time of writing and still marked as WIP, but it sounds like exactly what I want.

We can see here that GPT2 (and DistilGPT2) are supported: https://github.com/huggingface/exporters/blob/main/MODELS.md

Unfortunately we need the `GPT2LMHeadModel` for text generation, which is supported with caveat "no `use_past`". We can [see here](https://github.com/huggingface/exporters#exporting-a-decoder-model) that that corresponds to the `use_cache` arg in HF `GPT2Config`.

The docs say:

> **`use_cache`** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models).

Googling for what this means didn't turn up much. ChatGPT explained:

> If `use_cache` is set to `True`, the model will cache the previous hidden states and attention values generated during inference. This can significantly speed up inference on longer sequences since the model does not need to recompute the entire sequence for each forward pass. Instead, it only computes the new portion of the sequence that is being generated.
>
> However, if `use_cache` is set to `False`, the model will not use the cache during inference. This means that the model will need to recompute the entire sequence for each forward pass, which can be slower and more computationally expensive, especially for longer sequences.

This sounds to me like a 'forward pass' is the process of generating a new token.

If this plausible info is correct, and not a hallucination, then it suggests the model retains all its capabilities with `use_cache=False` but will run slower, increasingly so when generating longer outputs.

This looks like the way forward for now.

There is also: https://github.com/huggingface/optimum, which _"provides multiple tools to export and run optimized models on various ecosystems"_.  One of those is ONNX, which itself [provides CoreML support](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) ...and the `optimum` ONNX provider has an FP16 option, which I think was one of the prerequisites for running on ANE. Something to look into.

See also https://github.com/huggingface/exporters/issues/8

</details>

## Chapter 3

At this stage I have a basic PyTorch server working.

```
python -m experiment.server_pytorch --model=gpt2
```

We can see some timings to load different sizes of the model (on an M1 Macbook Air 16GB RAM):

| model | loading time |
|-|-:|
| `gpt2` ("small") |  2.7s |
| `gpt2-medium`    |  6.0s |
| `gpt2-large`     | 12.6s |
| `gpt2-xl`        | 29.5s |
| `distilgpt2`     |  1.8s |

Coincidentally, the time needed to generate a single response with each model seems to be roughly the same as its loading time. i.e. ~30s for `gpt2-xl`.

### Export to CoreML?

I tried this https://github.com/huggingface/exporters as GPT2 was listed as a supported model.

But I ran into this problem: https://github.com/huggingface/exporters/issues/18

So, that looks like a dead end for now.

### Next steps?

One option would be to try again with `swift-coreml-transformers`. Perhaps try to write a Tokenizer class for HF/Pytorch which mimics the behaviour of the Swift one. Get it into a usable state and then benchmark it.

Another would be to attempt a ground-up rewrite of GPT2 model, implementing the changes described in https://machinelearning.apple.com/research/neural-engine-transformers

i.e. do for GPT2 what Apple already did for DistilBERT:
https://github.com/apple/ml-ane-transformers/blob/main/ane_transformers/huggingface/distilbert.py

