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

### Update

Looks like someone else already did it (just in the last couple of weeks LOL):  
https://github.com/smpanaro/more-ane-transformers

Implements GPT2, based on https://github.com/karpathy/nanoGPT codebase, with the tweaks from `ane-transformers`.

The notes are interesting: https://github.com/smpanaro/more-ane-transformers/blob/main/src/experiments/NOTES.md

The tweaks they had to make are a little over my head and more extensive than I'd hoped. One thing they turned up seems to be a ~3GB memory limit, if the model is bigger it won't run on ANE. So that's approx 1.5B float16 model params. (@smpanaro seems to have got GPT2-XL (1.5B) running on the ANE.)

This would seem to end hopes of one day running LLaMA 7B on ANE. E.g. quantized to Int4 that's still 3.5GB and I think ANE only deals with float16 anyway. Another idea is to sparsify the model (see https://github.com/AlpinDale/sparsegpt-for-LLaMA) ...it seems like 50% reduction in params is possible for the larger variants, but less so for the smaller.

Or GPT-J-6B, reduced by 50% would be right at the limit and therefore likely too large.

Promising models to potentially convert and run on ANE:
- https://huggingface.co/bigcode/santacoder a 1.1B param GPT2 model, [trained beyond Chinchilla-optimal](https://www.harmdevries.com/post/model-size-vs-compute-overhead/), on Python, Javascript & Java code. Seems both useful (local Copilot?) and feasible on ANE!
- https://huggingface.co/EleutherAI/gpt-neo-1.3B It is a GPT-3 type which has better evaluation scores than GPT-2 1.5B (XL) and gives quite nice completions in the demo box on that page.
- https://huggingface.co/google/flan-t5-large has 770M (or 1.1B?) params. It won't be good for chat, but the T5 models do very well at e.g. classification tasks, especially with fine-tuning.
  - https://huggingface.co/Salesforce/codet5-large 770M "for code understanding and generation"
  - https://huggingface.co/Salesforce/codet5-large-ntp-py 770M
  - https://github.com/salesforce/CodeT5 the models above were subsequently trained via RL (on 'code challenge' word problem tasks), resulting in `CodeT5-finetuned_CodeRL` with checkpoint here: https://console.cloud.google.com/storage/browser/sfr-coderl-research/codet5_finetuned_codeRL;tab=objects?prefix=&forceOnObjectsSortingFiltering=false
- Facebook/Meta have https://huggingface.co/facebook/opt-1.3b and also https://huggingface.co/KoboldAI/fairseq-dense-1.3B (doesn't seem to have a 1st party version published to HF) ... ohhh "fair" is FAIR as in Facebook AI Research.
- https://huggingface.co/OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 Pythia-1.4B with a GPT4all fine-tuning. I tried [chatting](https://open-assistant.io/chat/) with the bigger `OA_SFT_Llama_30B` and it is poor compared to ChatGPT, unsurprisingly, so I wouldn't hold much hope for this.
- https://huggingface.co/bigscience/bloom-1b1 text generation with a science-y flavour
- https://github.com/salesforce/CodeGen has 350M and 2B versions... 2B is too big as-is but maybe it could be distilled/pruned to fit. Only does "code gen from a comment string" i.e. roughly like Copilot.
- https://mahis.life/bet/ / https://github.com/notmahi/bet Behaviour Transformers (for robot/agent control). Built on MinGPT (superseded by NanoGPT, as used by `more-ane-transformers` GPT2) don't know how many params. No weights released I think, have to train it yourself. Code and dataset are supplied though. Is something like this of any use for more textual agents?

In short, the kind of models you can run on the ANE are the smaller ones that you will want to fine-tune for a specific task. The LLMs that can generalize across tasks are likely all too big.

### Other resources

- https://github.com/NVIDIA/FasterTransformer/

    > FasterTransformer implements a highly optimized transformer layer for both the encoder and decoder for inference. On Volta, Turing and Ampere GPUs, the computing power of Tensor Cores are used automatically when the precision of the data and weights are FP16.

    It's a different architecture and probbaly some stuff is NVIDIA-specific. But I wonder if some aspects are transferable.

- https://github.com/Ki6an/fastT5

    Similarly.

- https://coremltools.readme.io/docs/compressing-ml-program-weights#use-sparse-representation

    `coremltools` has support for sparse weights and tool to sparsify them. Does this translate all the way through to memory usage of the model at inference time?  I'm guessing maybe yes, since most of the Apple tooling has an eye towards iPhone usage.
