import json
import re

# Load the TSV content
with open('data/hipe/HIPE-2022-v2.1-hipe2020-test-fr.tsv', 'r') as f:
    lines = f.readlines()

sentences = []
sentence = []
for line in lines:
    if "# " not in line and 'TOKEN\t' not in line:
        parts = line.strip().split("\t")
        sentence.append(parts)
        if len(parts) > 9 and "EndOfSentence" in parts[9]:
            sentences.append(sentence)
            sentence = []

# Load the JSON content
with open('data/results/model_meta-llama_Llama-2-7b-hf_peft_exp_hipe_llama7b_prompt_1_epoch'
                        '-2__max_new_tokens_100_seed_42_do_sample_True_min_length_None_use_cache_True_top_p_1'
                        '.0_temperature_1.0_max_padding_4096.jsonl', 'r') as f:
    json_lines = [json.loads(line) for line in f]


def replace_elements(data):
    for row in data:
        # Keep the first and last three elements
        to_keep = row[:3] + row[-3:]

        # Replace the rest with 'O'
        for i in range(3, len(row) - 3):
            row[i] = 'O'

    return data
# For each sentence in the TSV, map the tokens with the corresponding tags from the pred_tags
def remove_tags(text):
    text_no_tags = re.sub(r"<\/?.+?>", "", text)
    return re.sub(r"\s+", " ", text_no_tags).strip()  # Replacing consecutive spaces with a single space


for sentence, json_line in zip(sentences, json_lines):
    tokens = json_line['tokens'].split()
    pred_tags_str = json_line['pred_tags']
    ground_tokens = [line[0] for line in sentence]
    ground_sentence = ' '.join(ground_tokens)

    # This pattern matches the outer tags and contents.
    pattern_outer = r"<(.*?)>(.*?)<\/\1>"
    matches_outer = re.findall(pattern_outer, pred_tags_str)

    formatted_entities = []

    for match in matches_outer:
        tag_outer, text_outer = match

        # This pattern matches the inner tags and contents.
        pattern_inner = r"<(.*?)> (.*?) <\/\1>"
        matches_inner = re.findall(pattern_inner, text_outer)

        # Add the inner entities to the list.
        for inner_match in matches_inner:
            tag_inner, text_inner = inner_match
            text_inner = remove_tags(text_inner.strip())  # TODO: correct this!!!
            formatted_entities.append((text_inner.strip(), tag_inner))

            # Remove the inner tag from the outer text
            text_outer = text_outer.replace(f"<{tag_inner}> {text_inner} </{tag_inner}>".strip(), text_inner)

        text_outer = remove_tags(text_outer.strip()) # TODO: correct this!!!

        # if '<' in text_outer:
        #     print(tag_inner, text_inner)
        formatted_entities.append((text_outer.strip(), tag_outer))
        # Add the cleaned outer entity to the list
        # formatted_entities.append((text_outer.strip(), tag_outer))

    # print(formatted_entities)

    """
    sieur	B-pers	O	B-pers.ind	O	B-comp.name	O	NIL	_	_
    Daniel	I-pers	O	I-pers.ind	O	O	O	NIL	_	_
    Meuron	I-pers	O	I-pers.ind	O	B-comp.name	O	NIL	_	NoSpaceAfter
    ,	I-pers	O	I-pers.ind	O	O	O	NIL	_	_
    maïUe	I-pers	O	I-pers.ind	O	B-comp.function	O	NIL	_	_
    """
    pred_sentence = replace_elements(sentence)

    for entity in formatted_entities:
        entity_text, tag = entity
        if entity_text not in ground_sentence:
            print(entity_text, '--------', ground_sentence)

        entity_tokens = entity_text.split()
        entity_tags = [f'I-{tag}' for token in entity_tokens]
        entity_tags[0] = f'B-{tag}'

        # TODO: put the entity_tags in the right place in pred_sentence
        # if tag.count('.') > 1:

        # Find the correct placement in the sentence
        for i, word in enumerate(ground_tokens):
            if ground_tokens[i:i + len(entity_tokens)] == entity_tokens:
                for j, entity_tag in enumerate(entity_tags):
                    if 'comp' in entity_tag:
                        pred_sentence[i + j][5] = entity_tag
                    elif entity_tag.count('.') == 0:
                        pred_sentence[i + j][1] = entity_tag
                    elif entity_tag.count('.') == 1:
                        pred_sentence[i + j][3] = entity_tag
                    elif entity_tag.count('.') == 2:
                        pred_sentence[i + j][3] = entity_tag
                    elif 'time' in entity_tag:
                        if 'B-' in entity_tag:
                            pred_sentence[i + j][3] = 'B-time.date.abs'
                        else:
                            pred_sentence[i + j][3] = 'I-time.date.abs'
                    else:
                        print(entity_tag)
                    for type_tag in ['pers', 'org', 'loc']:
                        if type_tag in entity_tag:
                            limit = 6 if 'pers' in entity_tag else 5
                            pred_sentence[i + j][1] = entity_tag[:limit] #I-pers
                break
    # print(pred_sentence)


# Write the modified content to a new TSV file
with open('data/results/final_predictions.tsv', 'w') as out:
    out.write('TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC\n')
    for sentence in sentences:
        for parts in sentence:
            out.write('\t'.join(parts) + '\n')
            # print('\t'.join(parts))

print("Tags replaced successfully!")
