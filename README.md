# experiments-coreml-gpt2

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
