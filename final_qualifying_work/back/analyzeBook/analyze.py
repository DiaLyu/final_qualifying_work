import json
import pymorphy2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer, sent_tokenize, word_tokenize
from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)
from gensim.models import Word2Vec

from analyzeBook.syntaxParse import SyntaxParse
import pathlib
from pathlib import Path

class AnalyzeText:

    segmenter = Segmenter()
    morph_vocab = MorphVocab()

    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    ner_tagger = NewsNERTagger(emb)

    tokenizer=WordPunctTokenizer()
    morph = pymorphy2.MorphAnalyzer()

    def __init__ (self, file_path):
        self.file_path = file_path


    def analyze(self):
        # получаем данные из других файлов
        move_words = self.open_file_move()
        text_book = self.open_file_book()
        name_countries, city_counties = self.open_file_cities()

        doc = self.process_text(text_book)
        loc_spans, per_spans = self.get_names_cities(doc)

        normal_names = self.normalFormFIO(per_spans)

        # страны-города в тексте
        result_cities = self.get_cities_text(name_countries, city_counties, loc_spans)
        routes = self.get_routes(doc, result_cities, move_words, normal_names)
        self.descr_characters(doc, normal_names, routes)
        return routes


    def open_file_move(self):
        move_words = []
        path = Path(pathlib.Path.cwd(), 'analyzeBook', 'static', '2.txt')
        with open(path, encoding='utf-8') as movement:
            for line in movement:
                move_words = line.split()
        return move_words
    
    def open_file_book(self):
        lst = []
        text_lst = ""
        # ../books/Voina_i_mir.txt
        with open(self.file_path , encoding='utf-8') as text_book:
            for line in text_book:
                try:
                    line_text = line.strip()
                    if line_text != "":
                        lst.append(line_text)
                        text_lst += line_text + ' '
                except UnicodeEncodeError:
                    pass
        text_lst.replace('франц.', '')
        
        return text_lst
        
    def open_file_cities(self):
        path = Path(pathlib.Path.cwd(), 'analyzeBook', 'static', 'cities.json')
        with open(path, 'r', encoding='utf-8') as read_json:
            data_countries = json.load(read_json)

        name_countries = []         # названия стран
        city_counties = []          # список массивов городов по странам
        for obj in data_countries:
            country = [names.lower() for names in obj['country']['name']]
            name_countries.append(country)
            cities = []
            for obj_cit in obj['country']['cities']:
                for city in obj_cit:
                    cities.append(city.lower())
            city_counties.append(cities)

        for name in name_countries:
            name = list(map(lambda x: x.replace('ё','е'), name))

        for city in city_counties:
            city = list(map(lambda x: x.replace('ё','е'), city))

        return name_countries, city_counties

    def process_text(self, text):
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.parse_syntax(self.syntax_parser)
        doc.tag_ner(self.ner_tagger)

        for span in doc.spans:
            span.normalize(self.morph_vocab)

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)

        return doc

    def get_names_cities(self, doc):
        loc_spans = []
        per_spans = []
        for span in doc.spans:
            normal_word = span.normal.lower()
            if span.type == 'LOC':
                # извлечение из морфологического разбора слова существительное в именительном падеже
                morph_country = self.morph_vocab.lemmatize(normal_word, 'NOUN', {'Animacy': 'Inan', 'Case': 'Nom'})
                if morph_country.endswith(' гор'):
                    morph_country += "ы"
                loc_spans.append(morph_country)
            if span.type == 'PER':
                # print('PER ', span.tokens)
                tokens = span.tokens
                flg = True
                for token in tokens:
                    if token.pos != 'PROPN':
                        flg = False
                    if ('Animacy' in token.feats) and ('Number' in token.feats):
                        if token.feats['Animacy'] != 'Anim' and token.feats['Number'] != 'Sing':
                            flg = False
                if flg:     
                    morph_country = self.morph_vocab.lemmatize(normal_word, 'PROPN', {'Animacy': 'Anim', 'Case': 'Nom'})          
                    per_spans.append(morph_country)

        loc_spans = list(set(loc_spans))
        per_spans = list(set(per_spans))

        return loc_spans, per_spans
    

    # чистка текста и векторизация word2vec
    def descr_characters(self, doc, normal_names, routes):

        names = [nm_names['full_name'] for nm_names in normal_names]

        pos_sent = ['PUNCT', 'X', 'NUM', 'CCONJ']
        lemma_text = [[tokens.lemma for tokens in sents.tokens if tokens.pos not in pos_sent] for sents in doc.sents]

        text_stopwords = stopwords.words('russian') + stopwords.words('french') + stopwords.words('english')
        text_stopwords += ['ред', 'ну', 'ну-ка', 'ну-тка', 'ну-с', 'сюда', 'это', 'что','всё','сказать', 'говорить']
        for i in range(len(lemma_text)):
            lemma_text[i] = [w for w in lemma_text[i] if w not in text_stopwords]
        clear_text = [lem for lem in lemma_text if lem != []]

        lemma_text = []
        for i in range(len(clear_text)):
            sentence = []
            count_word = 0
            for ind in range(len(clear_text[i])):
                if count_word == 0:
                    if ind <= len(clear_text[i]) - 3:
                        nm = ' '.join(clear_text[i][ind:ind+3])
                        if nm in names:
                            sentence.append(nm)
                            count_word = 2
                        else:
                            sentence.append(clear_text[i][ind])
                    else:
                        sentence.append(clear_text[i][ind])
                else:
                    count_word -= 1
# триграммы
            sent_two = []
            count_word = 0
            for ind in range(len(sentence)):
                if count_word == 0:
                    if ind <= len(sentence) - 2:
                        nm = ' '.join(sentence[ind:ind+2])
                        if nm in names:
                            sent_two.append(nm)
                            count_word = 1
                        else:
                            sent_two.append(sentence[ind])
                    else:
                        sent_two.append(sentence[ind])
                else:
                    count_word -= 1
            lemma_text.append(sent_two)

        model = Word2Vec(window=5, min_count=3, vector_size = 100)

        # Train the model
        model.build_vocab(lemma_text, progress_per=1000)
        model.train(lemma_text, total_examples=model.corpus_count, epochs=model.epochs)

        k = 0
        for i in range(len(routes)):
            try:
                most_simil = model.wv.most_similar(routes[i]['Name'], topn=20)
                descr_char = ', '.join([similar[0] for similar in most_simil])
                routes[i]["Descr"] = descr_char
            except:
                routes[i]["Descr"] = ""


    # приведение к нормальной форме ФИО
    def normalFormFIO(self, per_spans):

        result_charact = []
        for pers in per_spans:
            list_key = []
            tokens = self.tokenizer.tokenize(pers)
            
            for tkn in tokens:
                p = self.morph.parse(tkn)
                
                cont_flag = False
                tags_name = ['Surn', 'Name', 'Patr']
                for pars in p:
                    split_tags = (' '.join(str(pars.tag).split(','))).split(' ')
                    tag = list(set(tags_name) & set(split_tags))
                    if len(tag) != 0:
                        list_key.append({'gend': pars.tag.gender, tag[0]: pars.normal_form})
                        cont_flag = True

                    if cont_flag:
                        break

            # в случаях, если повторяются имена, фамилии, отчества, или они не в стандартном порядке, то разделяем их
            if len(list_key) != 0:
                if len(list_key) == 3:
                    prop_1 = list(list_key[0])
                    prop_2 = list(list_key[1])
                    prop_3 = list(list_key[2])
                    if 'Name' in prop_1 and 'Patr' in prop_2 and 'Surn' in prop_3:
                        result_charact.append({'gend': list_key[0]['gend'], 'Name': list_key[0]['Name'], 'Patr': list_key[1]['Patr'], 'Surn': list_key[2]['Surn']})

                    elif 'Surn' in prop_1 and 'Name' in prop_2 and 'Patr' in prop_3:
                        result_charact.append({'gend': list_key[1]['gend'], 'Name': list_key[1]['Name'], 'Patr': list_key[2]['Patr'], 'Surn': list_key[0]['Surn']})

                    else:
                        if 'Name' in prop_1 and 'Patr' in prop_2:
                            result_charact.append({'gend': list_key[0]['gend'], 'Name': list_key[0]['Name'], 'Patr': list_key[1]['Patr']})
                            result_charact.append(list_key[2])

                        elif 'Name' in prop_2 and 'Patr' in prop_3:
                            result_charact.append({'gend': list_key[1]['gend'], 'Name': list_key[1]['Name'], 'Patr': list_key[2]['Patr']})
                            result_charact.append(list_key[0])

                        elif 'Name' in prop_1 and 'Surn' in prop_2:
                            result_charact.append({'gend': list_key[0]['gend'], 'Name': list_key[0]['Name'], 'Surn': list_key[1]['Surn']})
                            result_charact.append(list_key[2])

                        elif 'Name' in prop_2 and 'Surn' in prop_3:
                            result_charact.append({'gend': list_key[1]['gend'], 'Name': list_key[1]['Name'], 'Surn': list_key[2]['Surn']})
                            result_charact.append(list_key[0])

                elif len(list_key) == 2:
                    prop_1 = list(list_key[0])
                    prop_2 = list(list_key[1])
                    if 'Name' in prop_1 and 'Patr' in prop_2:
                        result_charact.append({'gend': list_key[0]['gend'], 'Name': list_key[0]['Name'], 'Patr': list_key[1]['Patr']})

                    elif 'Name' in prop_1 and 'Surn' in prop_2:
                        result_charact.append({'gend': list_key[0]['gend'], 'Name': list_key[0]['Name'], 'Surn': list_key[1]['Surn']})

                    else:
                        result_charact.append(list_key[0])
                        result_charact.append(list_key[1])

                else:
                    prop_1 = list(list_key[0])
                    result_charact.append({prop_1[0]: list_key[0][prop_1[0]], prop_1[1]: list_key[0][prop_1[1]]})

        normal_list = []

        for charact in result_charact:
            full_name = ""
            not_normal_name = ""
            for key in charact:
                if key != 'gend':
                    not_normal_name += charact[key] + " "

                    try:
                        parse_name = self.morph.parse(charact[key])[0].inflect({'sing', 'nomn', charact['gend']})
                    except:
                        parse_name = None

                    if not parse_name:
                        if (charact[key] == 'ростов' or charact[key] == 'ахросимов') and charact['gend'] == 'femn':
                            full_name += charact[key] + 'а '
                        elif charact[key] == 'василья':
                            full_name += 'василий '
                        elif charact[key] == 'амалье':
                            full_name += 'амалья '
                        else:
                            full_name += charact[key] + ' '

                    else:
                        if parse_name.word == "марь":
                            full_name += "марья "
                        elif parse_name.word == "анне":
                            full_name += "анна "
                        elif parse_name.word == "зуб":
                            full_name += "зубов "
                        else:
                            full_name += parse_name.word + ' '

            if len(full_name.strip()) > 1:
                normal_list.append({"full_name": full_name.strip(), "not_normal_name": not_normal_name.strip()})

        normal_list = list({v["not_normal_name"]:v for v in normal_list}.values())

        return normal_list
 
    
    def get_cities_text(self, name_countries, city_counties, loc_spans):
        result_cities = []
        for i in range(len(name_countries)):
            list_cities = set(city_counties[i]) & set(loc_spans)
            list_names = set(name_countries[i]) & set(loc_spans)
            result_cities.extend(list(list_cities))
            result_cities.extend(list(list_names))
        return result_cities

    def get_routes(self, doc, result_cities, move_words, normal_names):

        not_norm_name = [names["not_normal_name"] for names in normal_names]
        full_names = [names["full_name"] for names in normal_names]

        sents = doc.sents
        past_end_word = ['л', 'ла', 'ли', 'шел', 'лся', 'лись', 'лась', 'вшись', 'в', 'вшийся', 'ая']

        routes_hero = []

        for i in range(len(sents)):
            snt = sents[i]
            process = False

            index = 0
            for token in snt.tokens:
                if 'ROMN' in self.morph.parse(token.text)[0].tag:
                    snt.tokens.remove(token)

                elem = any([token.text.endswith(ends) for ends in past_end_word])
                for move in move_words:
                    if (token.lemma == move) and elem:
                        if index != 0:
                            if snt.tokens[index - 1].text != 'не':
                                process = True

                if 'возвращение' in token.lemma or 'приезд' in token.lemma:
                    process = True

                index += 1

            if 'в это время' in snt.text.lower() or 'в это же время' in snt.text.lower():
                process = True

            if process:
                tk = SyntaxParse(snt.tokens) 
                phrase = tk.phrases()
                not_names = ['тит', 'ата', 'поль', 'пол', 'ая', 'але', 'дава']
                for phr in phrase:
                    city = ' '.join([tkn.lemma for tkn in phr['City']])
                    for res in result_cities:
                        if res in city:
                            nameFIO = phr['Name']
                            nameFIO_join = ' '.join([token.lemma for token in nameFIO])
                            pron = [token for token in nameFIO if token.pos == 'PRON']
                            not_process_names = []

                            if pron == []:

                                for j in range(len(not_norm_name)):
                                    if not_norm_name[j] in nameFIO_join:
                                        not_process_names.append(full_names[j])

                                if len(not_process_names) == 0:
                                    list_name_full = self.normalFormFIO([' '.join([token.lemma for token in nameFIO])])
                                    not_process_names = [names["full_name"] for names in list_name_full]
                                
                                not_process_names = [item for item in not_process_names if item not in not_names]

                            else:
                                if i != 0:
                                    iter = 1
                                    while len(not_process_names) == 0 and i - iter >= 0:
                                        last_sent = ' '.join([token.lemma for token in sents[i-iter].tokens])
                                        for ind in range(len(not_norm_name)):
                                            if not_norm_name[ind] in last_sent and not_norm_name[ind] not in not_names:
                                                not_process_names.append(full_names[ind])
                                        iter += 1

                            if len(not_process_names) != 0:
                                for names in not_process_names:
                                    routes_hero.append({'Name': names, 'City': res})

        routes = []
        for route in routes_hero:
            flag = True
            for name_route in routes:
                if name_route['Name'] == route['Name']:
                    if name_route['Route'][len(name_route['Route']) - 1] != route['City']:
                        name_route['Route'].append(route['City'])
                    flag = False
            if flag:
                routes.append({'Name': route['Name'], 'Route': [route['City']]})     

        return routes
    

# analyzeText = AnalyzeText('../books/Voina_i_mir.txt')
# routes = analyzeText.analyze()
# print(routes)

# print("\n---------------\n")

# analyzeText = AnalyzeText('../books/lermontov1.txt')
# print(analyzeText.analyze())