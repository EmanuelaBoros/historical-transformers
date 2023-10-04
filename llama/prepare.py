import ssl
import argparse
import os
from utils.data_utils import _read_conll, get_entities
from tqdm import tqdm
from wikidata.client import Client
import logging
logging.basicConfig(level=logging.INFO)

ssl._create_default_https_context = ssl._create_unverified_context

client = Client()

COLUMNS = [
        "TOKEN",
        "NE-COARSE-LIT",
        "NE-COARSE-METO",
        "NE-FINE-LIT",
        "NE-FINE-METO",
        "NE-FINE-COMP",
        "NE-NESTED",
        "NEL-LIT",
        "NEL-METO",
        "MISC"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--lang",
        type=str,
        help="For Wikipedia",
    )
    parser.add_argument(
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args, _ = parser.parse_known_args()

    for root, dirs, files in os.walk(args.input_dir, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            if ('.tsv' in filename) \
                and ('masked' not in filename) \
                and ('results' not in filename) \
                and ('test' in filename) \
                and ('nel' not in filename) \
                and ('bart' not in filename):

                print(filename)
                with open(filename, 'r') as f:
                    lines = f.readlines()

                headers = [
                    'raw_words', 'target', 'link'
                ]
                # TODO: This needs to be changed if the data format is different or the
                # order of the elements in the file is different
                indexes = list(range(len(COLUMNS)))  # -3 is for EL

                if not isinstance(headers, (list, tuple)):
                    raise TypeError(
                        'invalid headers: {}, should be list of strings'.format(headers))
                phrases = _read_conll(
                    filename,
                    encoding='utf-8',
                    sep='\t',
                    indexes=indexes,
                    dropna=True)

                # GENRE requires the files to be named train.target and train.source
                type_of_file = 'train'
                if 'dev' in name:
                    type_of_file = 'dev'
                elif 'test' in name:
                    type_of_file = 'test'
                filename = os.path.join(args.output_dir, type_of_file + '.tsv')

                # import pdb;pdb.set_trace()
                sentences = []
                with open(filename.replace('.tsv', '.target'), 'w') as t_file:
                    with open(filename.replace('.tsv', '.source'), 'w') as s_file:
                        for phrase in tqdm(phrases, total=len(phrases)):

                            # [('R . Ellis', 'pers', 'Q7344037'), ('the Cambridge Journal of Philology', 'work', 'NIL'),
                            # ('Vol . IV', 'scope', 'NIL'), ('A . Nauck', 'pers', 'NIL'), ('Leipzig', 'loc', 'NIL'),
                            # ('1856', 'date', 'NIL')]

                            idx, phrase = phrase
                            tokens, entity_tags, link_tags = phrase[0], phrase[1], phrase[-3]
                            entities = get_entities(
                                tokens, entity_tags, link_tags)

                            # [('R . Ellis', 'pers', 'Q7344037', [12, 13, 14])
                            pos_qid, pos_ner = {}, {}
                            for entity in entities:
                                meta = {
                                    "left_context": ' '.join(tokens[:entity[-1][0]]),
                                    "right_context": ' '.join(tokens[entity[-1][-1] + 1:]),
                                    "mention": entity[0],
                                    "label_title": entity,
                                    "label": entity,
                                    "label_id": entity
                                }

                                qid = entity[2]

                                if qid != 'NIL':

                                    try:
                                        wikidata_entity = client.get(qid, load=True)
                                    except:
                                        logging.info(
                                            'HTTP Error 404: {} not Found. Replaced with NIL.'.format(qid))
                                        possible_entity = 'NIL'
                                    try:
                                        possible_entity = wikidata_entity.label[args.lang]
                                    except BaseException as ex:
                                        try:
                                            possible_entity = wikidata_entity.label[next(
                                                iter(wikidata_entity.label.texts))]
                                        except:
                                            possible_entity = entity[0]
                                        logging.info(
                                            'No text was found for {} in {}. Replaced with {}.'.format(
                                                qid, args.lang, possible_entity))
                                else:
                                    possible_entity = 'NIL'

                                paragraph = meta["left_context"] + " [START] " + meta["mention"] + " [END] " + meta["right_context"]
                                sentences.append(paragraph)
                                s_file.write(paragraph + '\n')
                                t_file.write(
                                    possible_entity + ' >> ' + args.lang + '\n')

                                s_file.flush()
                                t_file.flush()
