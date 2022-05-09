import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration

model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv")
tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")

ARTICLE_TO_SUMMARIZE = '''Transformers (Vaswani et al., 2017) have achieved state-of-the-art
    results in a wide range of natural language tasks including generative language modeling
    (Dai et al., 2019; Radford et al., 2019) and discriminative ... language understanding (Devlin et al., 2019).
    This success is partly due to the self-attention component which enables the network to capture contextual
    information from the entire sequence. While powerful, the memory and computational requirements of
    self-attention grow quadratically with sequence length, making it infeasible (or very expensive) to
    process long sequences. To address this limitation, we present Longformer, a modified Transformer
    architecture with a self-attention operation that scales linearly with the sequence length, making it
    versatile for processing long documents (Fig 1). This is an advantage for natural language tasks such as
    long document classification, question answering (QA), and coreference resolution, where existing approaches
    partition or shorten the long context into smaller sequences that fall within the typical 512 token limit
    of BERT-style pretrained models. Such partitioning could potentially result in loss of important
    cross-partition information, and to mitigate this problem, existing methods often rely on complex
    architectures to address such interactions. On the other hand, our proposed Longformer is able to build
    contextual representations of the entire context using multiple layers of attention, reducing the need for
    task-specific architectures.'''

ARTICLE_TO_SUMMARIZE_SLO = '''Rimsko cesarstvo je bilo obdobje starega Rima, ki je sledilo Rimski republiki. Kot država je obsegalo veliko ozemlje okoli Sredozemskega morja v Evropi, severni Afriki in zahodni Aziji. V cesarstvu so vladali cesarji. Od začetka vladavine cesarja Avgusta do vojaške anarhije v 3. stoletju je bila država principat z Italijo kot metropolo provinc in Rimom kot edino prestolnico (27 pr. n. št. - 286 n. št.). Po krizi 3. stoletja je bilo cesarstvo razdeljeno v Zahodno rimsko cesarstvo in Vzhodno rimsko cesarstvo. Slednje je znano tudi kot Bizantinsko cesarstvo. Cesarstvi sta imeli vsako svojega cesarja. Uradna prestolnica obeh cesarstev je do leta 476 ostal Rim. Tisto leto so Raveno zasedli Odoakerjevi Ostrogoti in odstavili zadnjega zahodnorimskega cesarja Romula Avgusta, zato so cesarske insignije prenesli v Konstantinopel. S sprejetjem krščanstva kot državne vere Rimskega cesarstva leta 380 in padcem Zahodnega rimskega cesarstva se je končalo obdobje klasične antike in začel srednji vek. Ti dogodki in postopna helenizacija Vzhodnega rimskega cesarstva so razlog, da zgodovinarji srednjeveško Rimsko cesarstvo, ki je ostalo v vzhodnih rimskih provincah, imenujejo Bizantinsko cesarstvo. '''

inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE_SLO, return_tensors="pt")

# Global attention on the first token (cf. Beltagy et al. 2020)
global_attention_mask = torch.zeros_like(inputs)
global_attention_mask[:, 0] = 1

# Generate Summary
summary_ids = model.generate(inputs, global_attention_mask=global_attention_mask, num_beams=3, max_length=64)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(f'summary: {summary}')