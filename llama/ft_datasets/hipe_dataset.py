# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


PROMPT_DICT = {
    "prompt_input_1": (
        """Below is an instruction that describes a task, paired with an input that provides further context.
        
        Instruction:
        Annotate the entities within the given text based on the guidelines provided. 
        Ensure the entities and their sub-components are accurately tagged. Once you've familiarized yourself with the guidelines below, you will be provided a text to annotate.
            
        Entity Guidelines:
        1. Entity Type: Person
            Subtypes:
                pers.ind: Individual persons.
                pers.ind.articleauthor: Author of a newspaper article.
                pers.coll: Groups of individuals identifiable by a proper name, e.g., the Beatles.
            Components:
                comp.func: Roles, including occupations, administrative roles, or societal roles.
                comp.title: Titles (Mr., Mrs., etc.), military ranks, noble, and royal titles.
                comp.qualifier: Adjectives describing the entity.
                comp.name: Parts of a name: first, middle, last, or nicknames.
                demonym: Residents of a specific location.
            Examples:
                Example: 
                    Mais il a cédé quand M . Hagenbeck a déclaré qu ' il prenait la responsabilité de ce qui pourrait arriver !" ➞ 
                Response: 
                    Mais il a cédé quand <pers.ind> <comp.title> M . </comp.title> <comp.name> Hagenbeck </comp.name> </pers.ind> a déclaré qu ' il prenait la responsabilité de ce qui pourrait arriver !
                
                Example: 
                    Cette nomination lui permet d ' entrer dans la Chambre des lords .
                Answer: 
                    Cette nomination lui permet d ' entrer dans la <org.adm> Chambre des lords </org.adm> .
                
        2. Entity Type: Organisation
            Subtypes:
                org.ent: Companies selling products or providing services (e.g., hospitals, sports clubs).
                org.adm: Organisations with a primary administrative role (e.g., town halls, ministries).
                org.ent.pressagency: Newspapers material like AFP, Reuters.
            Examples:
                Example:
                    En même temps que le congrès , différentes - ociétés ont eu leur séance - annuelle : les Templiers , la Ligue anti - alcoolique , les étudiants abstinents _ , etc .
                Answer:
                    En même temps que le congrès , différentes - ociétés ont eu leur séance - annuelle : les <org.ent> Templiers </org.ent> , la <org.ent> Ligue anti - alcoolique </org.ent> , les <org.ent> étudiants abstinents _ </org.ent> , etc .
                
        3. Entity Type: Location
            A. Administrative Locations:
                loc.adm.town: Cities, villages, districts, boroughs.
                    Example:
                        L ' assemblée générale de la Société suisse des imprimeurs , réunie samedi et dimanche à Frauenfeld , comptait de nombreux _ parfic - - pants .
                    Answer:
                        L ' assemblée générale de la <org.ent> Société suisse des imprimeurs </org.ent> , réunie samedi et dimanche à <loc.adm.town> Frauenfeld </loc.adm.town> , comptait de nombreux _ parfic - - pants .
                loc.adm.reg: Internal divisions within a country.
                    Example:
                        Le comité conservateur du canton de Fribourg a décidé de recommander aux électeurs de voter l ' article concernant les arts et métiers et l ' initiative contre l ' absinthe .
                    Answer:
                        Le comité conservateur du <loc.adm.reg> canton de <comp.name> Fribourg </comp.name> </loc.adm.reg> a décidé de recommander aux électeurs de voter l ' article concernant les arts et métiers et l ' initiative contre l ' absinthe .
                loc.adm.nat: Countries.
                    Example:
                        La commission demande que l ' on s ' occupe spécialement de la situation déplorable des Suisses en Russie .
                    Answer:
                        La commission demande que l ' on s ' occupe spécialement de la situation déplorable des Suisses en <loc.adm.nat> Russie </loc.adm.nat> .
                loc.adm.sup: Global regions, continents.
                    Example:
                        Si l ' Angleterre parvient a mettre un terme , définitivement , au pelil jeu des rivalités stériles auquel trop longtemps elle se livra vis - à - vis de la France , en Méditerranée ou au Moyen - Oriertl , ce ne sera pas là un des moindres bienfaits à mettre à l ' actif de la transformation de régime qui s ' opère chez nos voisins d ' outre - Doubs .
                    Answer:
                        Si l ' <loc.adm.nat> Angleterre </loc.adm.nat> parvient a mettre un terme , définitivement , au pelil jeu des rivalités stériles auquel trop longtemps elle se livra vis - à - vis de la <loc.adm.nat> France </loc.adm.nat> , en <loc.adm.sup> Méditerranée </loc.adm.sup> ou au <loc.adm.sup> Moyen - Oriertl </loc.adm.sup> , ce ne sera pas là un des moindres bienfaits à mettre à l ' actif de la transformation de régime qui s ' opère chez nos voisins d ' <loc.adm.reg> outre - Doubs </loc.adm.reg> .
            B. Physical Places:
                loc.phys.geo: Terrestrial locations.
                    Example:
                        Les habitants ressemblent à ceux de la Forét - Noire , ne comprennent pas le français , qui n ' est enseigné que depuis vingt ans .
                    Answer:
                        Les habitants ressemblent à ceux de la <loc.phys.geo> Forét - Noire </loc.phys.geo> , ne comprennent pas le français , qui n ' est enseigné que depuis vingt ans .
                loc.phys.hydro: Aquatic sites.
                    Example:
                        Au nord du Danube et de la Save se trouve une grande quantité de butin et de bétail - que les Austro - Allemands ont emportée de ïa Serbie et qui sera ramenée au pays .
                    Answer:
                        Au nord du <loc.phys.hydro> Danube </loc.phys.hydro> et de la <loc.phys.hydro> Save </loc.phys.hydro> se trouve une grande quantité de butin et de bétail - que les Austro - Allemands ont emportée de ïa <loc.adm.nat> Serbie </loc.adm.nat> et qui sera ramenée au pays .
                loc.phys.astro: Astronomical places.
                    Example:
                        La planète Mars"
                    Answer:
                        <loc.phys.astro> La planète Mars </loc.phys.astro>
        
        4. Entity Type: Human Productions
            A. Media Production: prod.media
            Example:
                Nous recevons le premier numéro d ' un nouveau journal , le Radical - Libéral , qui paraîtra à Genève deux fois la semaine .
            Answer:
                Nous recevons le premier numéro d ' un nouveau journal , le <prod.media> Radical - Libéral </prod.media> , qui paraîtra à <loc.adm.town> Genève </loc.adm.town> deux fois la semaine .    
            B. Doctrine: prod.doctr
            Example:
                Son but est de représenter l ' élément national du radicalisme genevois , en d ' autres termes , de défendre la politique intransigeante do M . Carteret , en opposition aux tendances du groupe _ > dont le Genevois est l ' organe .
            Answer:
                Son but est de représenter l ' élément national du <prod.doctr> <comp.name> radicalisme </comp.name> genevois </prod.doctr> , en d ' autres termes , de défendre la politique intransigeante do <pers.ind> <comp.title> M . </comp.title> <comp.name> Carteret </comp.name> </pers.ind> , en opposition aux tendances du groupe _ > dont le Genevois est l ' organe .
        
        5. Entity Type: Time
            A. Date: time.date
            Example: 
                Il y a eu à Boncourt en 1886 : 24 naissances , dont 11 garçons , 13 filles , 11 bourgeois , 13 non bourgeois , 16 Suisses , 8 étrangers à la Suisse ; — 24 décès , dont 11 hommes , 13 femmes , 11 bourgeois , 13 non bourgeois , 16 Suisses , 8 étrangers à la Suisse .
            Answer:
                Il y a eu à <loc.adm.town> Boncourt </loc.adm.town> <time.date.abs> en 1886 </time.date.abs> : 24 naissances , dont 11 garçons , 13 filles , 11 bourgeois , 13 non bourgeois , 16 Suisses , 8 étrangers à la <loc.adm.nat> <loc.adm.nat> Suisse </loc.adm.nat> </loc.adm.nat> ; — 24 décès , dont 11 hommes , 13 femmes , 11 bourgeois , 13 non bourgeois , 16 Suisses , 8 étrangers à la Suisse .
        
        Please proceed with annotating the entities in the following text:
        Example:
        {tokens}
            
        Answer: 
            """
    ),
    "prompt_input_2": (
        """Below is an instruction that describes a task, paired with an input that provides further context.

        Instruction:
        Annotate the entities within the given text based on the guidelines provided by generating lists of annotated entities.
        Ensure the entities and their sub-components are accurately tagged. Once you've familiarized yourself with the guidelines below, you will be provided a text to annotate.

        Entity Guidelines:
        1. Entity Type: Person
            Subtypes:
                pers.ind: Individual persons.
                pers.ind.articleauthor: Author of a newspaper article.
                pers.coll: Groups of individuals identifiable by a proper name, e.g., the Beatles.
            Components:
                comp.func: Roles, including occupations, administrative roles, or societal roles.
                comp.title: Titles (Mr., Mrs., etc.), military ranks, noble, and royal titles.
                comp.qualifier: Adjectives describing the entity.
                comp.name: Parts of a name: first, middle, last, or nicknames.
                demonym: Residents of a specific location.
            Examples:
                Example: 
                    Mais il a cédé quand M . Hagenbeck a déclaré qu ' il prenait la responsabilité de ce qui pourrait arriver !" ➞ 
                Response: 
                    <pers.ind> <comp.title> M . </comp.title> <comp.name> Hagenbeck </comp.name> </pers.ind>

                Example: 
                    Cette nomination lui permet d ' entrer dans la Chambre des lords .
                Answer: 
                    Cette nomination lui permet d ' entrer dans la <org.adm> Chambre des lords </org.adm> .

        2. Entity Type: Organisation
            Subtypes:
                org.ent: Companies selling products or providing services (e.g., hospitals, sports clubs).
                org.adm: Organisations with a primary administrative role (e.g., town halls, ministries).
                org.ent.pressagency: Newspapers material like AFP, Reuters.
            Examples:
                Example:
                    En même temps que le congrès , différentes - ociétés ont eu leur séance - annuelle : les Templiers , la Ligue anti - alcoolique , les étudiants abstinents _ , etc .
                Answer:
                    <org.ent> Templiers </org.ent>, <org.ent> Ligue anti - alcoolique </org.ent>, <org.ent> étudiants abstinents _ </org.ent>

        3. Entity Type: Location
            A. Administrative Locations:
                loc.adm.town: Cities, villages, districts, boroughs.
                    Example:
                        L ' assemblée générale de la Société suisse des imprimeurs , réunie samedi et dimanche à Frauenfeld , comptait de nombreux _ parfic - - pants .
                    Answer:
                        <org.ent> Société suisse des imprimeurs </org.ent>, <loc.adm.town> Frauenfeld </loc.adm.town>
                loc.adm.reg: Internal divisions within a country.
                    Example:
                        Le comité conservateur du canton de Fribourg a décidé de recommander aux électeurs de voter l ' article concernant les arts et métiers et l ' initiative contre l ' absinthe .
                    Answer:
                        <loc.adm.reg> canton de <comp.name> Fribourg </comp.name> </loc.adm.reg>
                loc.adm.nat: Countries.
                    Example:
                        La commission demande que l ' on s ' occupe spécialement de la situation déplorable des Suisses en Russie .
                    Answer:
                        <loc.adm.nat> Russie </loc.adm.nat>
                loc.adm.sup: Global regions, continents.
                    Example:
                        Si l ' Angleterre parvient a mettre un terme , définitivement , au pelil jeu des rivalités stériles auquel trop longtemps elle se livra vis - à - vis de la France , en Méditerranée ou au Moyen - Oriertl , ce ne sera pas là un des moindres bienfaits à mettre à l ' actif de la transformation de régime qui s ' opère chez nos voisins d ' outre - Doubs .
                    Answer:
                        <loc.adm.nat> Angleterre </loc.adm.nat>, <loc.adm.nat> France </loc.adm.nat>, <loc.adm.sup> Méditerranée </loc.adm.sup>, <loc.adm.sup> Moyen - Oriertl </loc.adm.sup>, <loc.adm.reg> outre - Doubs </loc.adm.reg>
            B. Physical Places:
                loc.phys.geo: Terrestrial locations.
                    Example:
                        Les habitants ressemblent à ceux de la Forét - Noire , ne comprennent pas le français , qui n ' est enseigné que depuis vingt ans .
                    Answer:
                        <loc.phys.geo> Forét - Noire </loc.phys.geo>
                loc.phys.hydro: Aquatic sites.
                    Example:
                        Au nord du Danube et de la Save se trouve une grande quantité de butin et de bétail - que les Austro - Allemands ont emportée de ïa Serbie et qui sera ramenée au pays .
                    Answer:
                        <loc.phys.hydro> Danube </loc.phys.hydro>, <loc.phys.hydro> Save </loc.phys.hydro>, <loc.adm.nat> Serbie </loc.adm.nat>
                loc.phys.astro: Astronomical places.
                    Example:
                        La planète Mars"
                    Answer:
                        <loc.phys.astro> La planète Mars </loc.phys.astro>

        4. Entity Type: Human Productions
            A. Media Production: prod.media
            Example:
                Nous recevons le premier numéro d ' un nouveau journal , le Radical - Libéral , qui paraîtra à Genève deux fois la semaine .
            Answer:
                <prod.media> Radical - Libéral </prod.media>, <loc.adm.town> Genève </loc.adm.town>
            B. Doctrine: prod.doctr
            Example:
                Son but est de représenter l ' élément national du radicalisme genevois , en d ' autres termes , de défendre la politique intransigeante do M . Carteret , en opposition aux tendances du groupe _ > dont le Genevois est l ' organe .
            Answer:
                <prod.doctr> <comp.name> radicalisme </comp.name> genevois </prod.doctr>, <pers.ind> <comp.title> M . </comp.title> <comp.name> Carteret </comp.name> </pers.ind>

        5. Entity Type: Time
            A. Date: time.date
            Example: 
                Il y a eu à Boncourt en 1886 : 24 naissances , dont 11 garçons , 13 filles , 11 bourgeois , 13 non bourgeois , 16 Suisses , 8 étrangers à la Suisse ; — 24 décès , dont 11 hommes , 13 femmes , 11 bourgeois , 13 non bourgeois , 16 Suisses , 8 étrangers à la Suisse .
            Answer:
                <loc.adm.town> Boncourt </loc.adm.town>, <time.date.abs> en 1886 </time.date.abs>, <loc.adm.nat> <loc.adm.nat> Suisse </loc.adm.nat> </loc.adm.nat>

        Please proceed with annotating the entities in the following text:
        Example:
        {tokens}

        Answer: 
            """
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

import jsonlines
class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):

        data_train = []
        for input_file in ['data/hipe/HIPE-2022-v2.1-hipe2020-train-fr_universal.jsonl']:
            with jsonlines.open(input_file, 'r') as f:
                for line in f:
                    data_train.append(line)

        data_dev = []
        for input_file in ['data/hipe/HIPE-2022-v2.1-hipe2020-dev-fr_universal.jsonl']:
            with jsonlines.open(input_file, 'r') as f:
                for line in f:
                    data_dev.append(line)

        # self.ann = json.load(open(dataset_config.data_path))

        if partition == "train":
            self.ann = data_train
        else:
            self.ann = data_dev

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        if ann.get("tokens", "") == "":
            prompt = PROMPT_DICT["prompt_input_2"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input_2"].format_map(ann)

        example = prompt + ann["tags"]

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask
        }
