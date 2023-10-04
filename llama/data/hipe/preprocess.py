from nltk.chunk import conlltags2tree
from nltk import pos_tag
from nltk.tree import Tree
import jsonlines

data = """
Le	O	O	O	O	O	O	_	_	_
public	O	O	O	O	O	O	_	_	_
est	O	O	O	O	O	O	_	_	_
averti	O	O	O	O	O	O	_	_	_
que	O	O	O	O	O	O	_	_	_
Charlotte	B-pers	O	B-pers.ind	O	O	O	NIL	_	_
née	I-pers	O	I-pers.ind	O	O	O	NIL	_	_
Bourgoin	I-pers	O	I-pers.ind	O	B-comp.name	O	NIL	_	NoSpaceAfter
,	O	O	O	O	O	O	_	_	EndOfLine
femme	O	O	O	O	O	O	_	_	NoSpaceAfter
-	O	O	O	O	O	O	_	_	NoSpaceAfter
de	O	O	O	O	O	O	_	_	_
Joseph	B-pers	O	B-pers.ind	O	O	O	NIL	_	_
Digiez	I-pers	O	I-pers.ind	O	B-comp.name	O	NIL	_	NoSpaceAfter
,	O	O	O	O	O	O	_	_	_
et	O	O	O	O	O	O	_	_	_
Maurice	B-pers	O	B-pers.ind	O	O	O	NIL	_	_
Bourgoin	I-pers	O	I-pers.ind	O	B-comp.name	O	NIL	_	NoSpaceAfter
,	O	O	O	O	O	O	_	_	_
enfant	O	O	O	O	O	O	_	_	_
mineur	O	O	O	O	O	O	_	_	EndOfLine
représenté	O	O	O	O	O	O	_	_	_
par	O	O	O	O	O	O	_	_	_
le	O	O	O	O	O	O	_	_	_
sieur	B-pers	O	B-pers.ind	O	B-comp.title	O	NIL	_	_
Jaques	I-pers	O	I-pers.ind	O	O	O	NIL	_	_
Charles	I-pers	O	I-pers.ind	O	O	O	NIL	_	_
Gicot	I-pers	O	I-pers.ind	O	B-comp.name	O	NIL	_	_
son	O	O	O	O	O	O	_	_	_
curateur	O	O	O	O	O	O	_	_	NoSpaceAfter
,	O	O	O	O	O	O	_	_	EndOfLine
"""

def get_entities(tokens, tags):
    tags = [tag.replace('S-', 'B-').replace('E-', 'I-') for tag in tags]
    pos_tags = [pos for token, pos in pos_tag(tokens)]

    conlltags = [(token, pos, tg)
                 for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)

    entities = []
    idx = 0
    char_position = 0  # This will hold the current character position

    for subtree in ne_tree:
        # skipping 'O' tags
        if isinstance(subtree, Tree):
            original_label = subtree.label()
            original_string = " ".join(
                [token for token, pos in subtree.leaves()])

            entity_start_position = char_position
            entity_end_position = entity_start_position + len(original_string)

            entities.append(
                (original_string.split(),
                 original_label,
                 (idx,
                  idx + len(subtree)),
                    (entity_start_position,
                     entity_end_position)))
            idx += len(subtree)

            # Update the current character position
            # We add the length of the original string + 1 (for the space)
            char_position += len(original_string) + 1
        else:
            token, pos = subtree
            # If it's not a named entity, we still need to update the character
            # position
            char_position += len(token) + 1  # We add 1 for the space
            idx += 1

    return entities


def replace_sublist(original, to_replace, replacement):
    """
    Replaces a sublist (to_replace) in the original list with another list (replacement).

    Parameters:
    - original: The original list.
    - to_replace: The sublist that needs to be replaced.
    - replacement: The sublist with which to_replace should be replaced.

    Returns:
    - A new list with the sublist replaced. If to_replace is not found, the original list is returned.
    """
    if not to_replace:  # Empty list, return original
        return original

    for i in range(len(original) - len(to_replace) + 1):
        if original[i:i + len(to_replace)] == to_replace:
            return original[:i] + replacement + original[i + len(to_replace):]

    return original  # if to_replace not found in original

def generate_annotated_text(input_file, output_file):

    with open(input_file, 'r') as f:
        data = f.read()

    lines = data.strip().split("\n")
    output_tokens = []

    tokens, words, lit_entities, comp_entities = [], [], [], []
    with jsonlines.open(output_file, 'w') as f:
        for idx, line in enumerate(lines):
            line_tags = line.split("\t")

            if not line.startswith("# ") and not line.startswith("TOKEN	") and len(line.strip()) > 0:
                word = line_tags[0]

                ne_fine_lit = line_tags[3]
                ne_fine_comp = line_tags[5]

                lit_entities.append(ne_fine_lit)
                comp_entities.append(ne_fine_comp)
                words.append(word)
                tokens.append(word)

                end_of_sentence = 'EndOfSentence' in line_tags[-1]
                if end_of_sentence:

                    lit_entities = get_entities(words, lit_entities)
                    comp_entities = get_entities(words, comp_entities)

                    for entity in lit_entities:
                        words = replace_sublist(words, entity[0], [f"<{entity[1]}>"] + entity[0] + [f"</{entity[1]}>"])
                    for entity in comp_entities:
                        words = replace_sublist(words, entity[0], [f"<{entity[1]}>"] + entity[0] + [f"</{entity[1]}>"])

                    jsonline = {'tokens': ' '.join(tokens), 'tags': ' '.join(words)}
                    f.write(jsonline)

                    tokens, words, lit_entities, comp_entities = [], [], [], []

for input_file in ['HIPE-2022-v2.1-hipe2020-test-fr.tsv', 'HIPE-2022-v2.1-hipe2020-dev-fr.tsv', 'HIPE-2022-v2.1-hipe2020-train-fr.tsv']:
    generate_annotated_text(input_file, input_file.replace('.tsv', '_universal.jsonl'))