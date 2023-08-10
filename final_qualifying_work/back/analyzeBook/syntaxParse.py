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

# синтаксические правила для вывода строк с связанными именами и городами
class SyntaxParse:

     def __init__(self, syntaxStructure):
          self.syntaxStructure = syntaxStructure
          # self.headIdList = [token.head_id for token in syntaxStructure]

     def phrases(self):
          tokens = self.syntaxStructure

          connect_words = []
          for i in range(len(tokens)):
               properties = [{"id": tokens[j].id, "rel": tokens[j].rel, "index": j} for j in range(len(tokens)) if tokens[j].head_id == tokens[i].id]
               connect_words.append(properties)

          phrases = self.rule_13(connect_words, "root", "obl", "case", "obl", "case", "conj", "nsubj", "appos", "flat:name") # 9
          if not phrases:
               phrases = self.rule_15(connect_words, "root", "nsubj", "appos", "conj", "conj", "obl", "amod", "case")        # 8
          if not phrases:
               phrases = self.rule_16(connect_words, "root", "nsubj", "appos", "conj", "obl", "amod", "case")                # 7

          if not phrases:
               phrases = self.rule_12(connect_words, "root", "iobj", "obl", "case", "conj", "obl", "case")                   # 7
               # print("---2")

          if not phrases:
               phrases = self.rule_14(connect_words, "root", "nsubj", "acl", "obl", "case", "obl", "nmod")                   # 7
               # print("---3")

          if not phrases:
               phrases = self.rule_5(connect_words, "root", "advcl", "obl", "case", "nsubj", "flat:name")                    # 6
               # print("---4")

          if not phrases:
               phrases = self.rule_6(connect_words, "root", "nsubj", "appos", "flat:name", "obl", "case")                    # 6
               # print("---5")

          if not phrases:
               phrases = self.rule_5(connect_words, "root", "advcl", "obl", "case", "nsubj:pass", "flat:name")               # 6
               # print("---6")
               
          if not phrases:
               phrases = self.rule_7(connect_words, "root", "nsubj", "case", "acl:relcl", "advmod", "nsubj")                 # 6
               # print("---7")
               
          if not phrases:
               phrases = self.rule_8(connect_words, "root", "obj", "obl", "case", "nsubj", "appos")                          # 6
               # print("---8")
               
          if not phrases:
               phrases = self.rule_9(connect_words, "root", "nsubj", "obl", "case", "conj", "case")                          # 6
               # print("---9")

          if not phrases:
               phrases = self.rule_17(connect_words, "root", "nsubj", "obj", "amod", "case")                                  # 5

          if not phrases:
               phrases = self.rule_3(connect_words, "root", "nsubj", "obj", "acl:relcl", "case")                             # 5
               # print("---10")

          if not phrases:
               phrases = self.rule_3(connect_words, "ccomp", "nsubj", "advcl", "obl", "case")                                # 5
               # print("---10_1")

          if not phrases:
               phrases = self.rule_3(connect_words, "root", "nsubj", "conj", "obl", "case")                                  # 5
               # print("---11")
                         
          if not phrases:
               phrases = self.rule_3(connect_words, "root", "nsubj", "advcl", "obl", "case")                                 # 5
               # print("---12")

          if not phrases:
               phrases = self.rule_4(connect_words, "root", "obl", "case", "nmod", "flat:name")                              # 5 
               # print("---13") 

          if not phrases:
               phrases = self.rule_1(connect_words, "root", "nsubj", "flat:name", "obl", "case")                             # 5
               # print("---14")

          if not phrases:
               phrases = self.rule_1(connect_words, "root", "nsubj", "nmod", "obl", "case")                                  # 5
               # print("---15")

          if not phrases:
               phrases = self.rule_1(connect_words, "root", "nsubj", "appos", "obl", "case")                                 # 5
               # print("---16")
               # 
          if not phrases:
               phrases = self.rule_11(connect_words, "obl", "case", "nmod", "nmod", "case")                                  # 5
               # print("---17")

          if not phrases:
               phrases = self.rule_2(connect_words, "advcl", "nsubj", "obl", "case")                                         # 4
               # print("---18")

          if not phrases:
               phrases = self.rule_2(connect_words, "root", "nsubj", "obl", "case")                                          # 4
               # print("---19")

          if not phrases:
               phrases = self.rule_2(connect_words, "root", "obj", "obl", "case")                                            # 4
               # print("---20")

          if not phrases:
               phrases = self.rule_2(connect_words, "ccomp", "nsubj", "obj", "det")                                          # 4 +
               # print("---21")
               
          if not phrases:
               phrases = self.rule_2(connect_words, "root", "nsubj", "acl:relcl", "case")                                    # 4 +
               # print("---22")
                              
          if not phrases:
               phrases = self.rule_10(connect_words,"acl", "obl", "case", "nsubj")                                           # 4 +
               # print("---23")

          # for phrase in phrases:
          #      str_phrase = "Name: "
          #      for token in phrase['Name']:
          #           str_phrase += token.text + " "
          #      str_phrase += "City: "
          #      for token in phrase['City']:
          #           str_phrase += token.text + " "
          #      all_phrases.append(str_phrase)
          return phrases

     # self.rule_1 = ["root", ["nsubj", "flat:name"], ["obl", "case"]]
     # self.rule_6 = ["root", ["nsubj", "nmod"], ["obl", "case"]]
     # self.rule_10 = ["root", ["nsubj", "appos"], ["obl", "case"]]
     def rule_1(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure

          for i in range(len(tokens)):
               is_rule_1 = [False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    cityHero = []

                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              nameHero.append(tokens[property["index"]])
                              for nam in connect_words[property["index"]]:
                                   if nam['rel'] == rels[2]:
                                        nameHero.append(tokens[nam["index"]])  
                                        nameHero = sorted(nameHero, key=lambda x: x.start)
                                        is_rule_1[0] = True


                         if property['rel'] == rels[3]:
                              cityHero.append(tokens[property["index"]])
                              for case in connect_words[property["index"]]:
                                   if case['rel'] == rels[4]:
                                        cityHero.append(tokens[case["index"]])  
                                        cityHero = sorted(cityHero, key=lambda x: x.start)
                                        is_rule_1[1] = True
                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         phrases.append(name_city)
          return phrases
     

     # self.rule_2 = ["advcl", ["nsubj"], ["obl", "case"]]
     # self.rule_3 = ["root", ["nsubj"], ["obl", "case"]]
     # self.rule_5 = ["root", ["obj"], ["obl", "case"]]
     # --- 22 ["root", ["nsubj"], ["acl:relcl", "case"]]
     # --- 35 ["ccomp", ["nsubj"], ["obj", "det"]]
     def rule_2(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure

          for i in range(len(tokens)):
               is_rule_1 = [False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              nameHero.append(tokens[property["index"]])
                              is_rule_1[0] = True

                         if property['rel'] == rels[2]:
                              cityHero.append(tokens[property["index"]])
                              for case in connect_words[property["index"]]:
                                   if case['rel'] == rels[3]:
                                        cityHero.append(tokens[case["index"]])
                                        cityHero = sorted(cityHero, key=lambda x: x.start)
                                        is_rule_1[1] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         phrases.append(name_city)
          return phrases
     

     # self.rule_7 = ["root", 
     #                        ["nsubj"], 
     #                        ["conj", "obl", "case"]]
     # self.rule_8 = ["ccomp", 
     #                        ["nsubj"], 
     #                        ["advcl", "obl", "case"]]
     # --- 27 ["root", 
     #                   ["nsubj"], 
     #                   ["advcl", "obl", "case"]]
     def rule_3(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure

          for i in range(len(tokens)):
               is_rule_1 = [False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              nameHero.append(tokens[property["index"]]) 
                              is_rule_1[0] = True

                         if property['rel'] == rels[2]:
                              for obl in connect_words[property["index"]]:
                                   if obl['rel'] == rels[3]:
                                        cityHero.append(tokens[obl["index"]])  
                                        for case in connect_words[obl["index"]]:
                                             if case['rel'] == rels[4]:
                                                  cityHero.append(tokens[case["index"]])
                                                  cityHero = sorted(cityHero, key=lambda x: x.start)
                                                  is_rule_1[1] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         phrases.append(name_city)
          return phrases


     # self.rule_4 = ["root", 
     #                        ["obl", 
     #                             ["case"], 
     #                             ["nmod", "flat:name"]], 
     #                        ["obl", "case"]]
     def rule_4(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure

          for i in range(len(tokens)):
               is_rule_1 = [False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              cityHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[2]:
                                        cityHero.append(tokens[property_1["index"]])
                                        cityHero = sorted(cityHero, key=lambda x: x.start)
                                        is_rule_1[0] = True

                                   if property_1['rel'] == rels[3]:
                                        for nam in connect_words[property_1["index"]]:
                                             if nam['rel'] == rels[4]:
                                                  nameHero.append(tokens[nam["index"]])
                                                  is_rule_1[1] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         phrases.append(name_city)
          return phrases
     

     # self.rule_9 = ["root", 
     #                        ["advcl", "obl", "case"], 
     #                        ["nsubj", "flat:name"]]
     # --- 23 ["root", 
     #              ["advcl", "obl", "case"], 
     #              ["nsubj:pass", "flat:name"]]
     def rule_5(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure

          for i in range(len(tokens)):
               is_rule_1 = [False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              for obl in connect_words[property["index"]]:
                                   if obl['rel'] == rels[2]:
                                        cityHero.append(tokens[obl["index"]])  
                                        for case in connect_words[obl["index"]]:
                                             if case['rel'] == rels[3]:
                                                  cityHero.append(tokens[case["index"]])
                                                  cityHero = sorted(cityHero, key=lambda x: x.start)
                                                  is_rule_1[0] = True

                         if property['rel'] == rels[4]:
                              nameHero.append(tokens[property["index"]])
                              for nam in connect_words[property["index"]]:
                                   if nam['rel'] == rels[5]:
                                        nameHero.append(tokens[nam["index"]])  
                                        nameHero = sorted(nameHero, key=lambda x: x.start)
                                        is_rule_1[1] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         phrases.append(name_city)
          return phrases
     
     # ["root", 
     #         ["nsubj", "appos", 
     #                        ["flat:name"], 
     #                        ["flat:name"]], 
     #         ["obl", "case"]]
     def rule_6(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure

          for i in range(len(tokens)):
               is_rule_1 = [False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              for obl in connect_words[property["index"]]:
                                   if obl['rel'] == rels[2]:
                                        nameHero.append(tokens[obl["index"]])  
                                        names = connect_words[obl["index"]]
                                        for j in range(len(names)):
                                             if j != len(names) - 1:
                                                  if names[j]['rel'] == rels[3] and names[j+1]['rel'] == rels[3]:
                                                       nameHero.append(tokens[names[j]["index"]])
                                                       nameHero.append(tokens[names[j + 1]["index"]])
                                                       nameHero = sorted(nameHero, key=lambda x: x.start)
                                                       is_rule_1[0] = True

                         if property['rel'] == rels[4]:
                              cityHero.append(tokens[property["index"]])
                              for nam in connect_words[property["index"]]:
                                   if nam['rel'] == rels[5]:
                                        cityHero.append(tokens[nam["index"]])
                                        cityHero = sorted(cityHero, key=lambda x: x.start)
                                        is_rule_1[1] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         phrases.append(name_city)
          return phrases
     
     # --- 24 ["root", 
     #              ["nsubj"],
     #              ["case"], 
     #              ["acl:relcl", 
     #                        ["advmod"], 
     #                        ["nsubj"] ]]
     def rule_7(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure
          for i in range(len(tokens)):
               is_rule_1 = [False, False, False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              nameHero.append(tokens[property["index"]])
                              is_rule_1[0] = True

                         if property['rel'] == rels[2]:
                              cityHero.append(tokens[property["index"]])
                              cityHero.append(tokens[i])
                              is_rule_1[1] = True

                         if property['rel'] == rels[3]:
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[4]:
                                        is_rule_1[2] = True
                                   if property_1['rel'] == rels[5]:
                                        is_rule_1[3] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         phrases.append(name_city)
          return phrases


     # --- 25 ["root", 
     #              ["obj"], 
     #              ["obl", "case"], 
     #              ["nsubj", "appos"]]
     def rule_8(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure
          for i in range(len(tokens)):
               is_rule_1 = [False, False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    nameSecHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              nameHero.append(tokens[property["index"]])
                              is_rule_1[0] = True

                         if property['rel'] == rels[2]:
                              cityHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[3]:
                                        cityHero.append(tokens[property_1["index"]])
                                        cityHero = sorted(cityHero, key=lambda x: x.start)
                                        is_rule_1[1] = True

                         if property['rel'] == rels[4]:
                              nameSecHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[5]:
                                        nameSecHero.append(tokens[property_1["index"]])
                                        nameSecHero = sorted(nameSecHero, key=lambda x: x.start)
                                        is_rule_1[2] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         name_sec_city = {"Name": nameSecHero, "City": cityHero}
                         phrases.append(name_city)
                         phrases.append(name_sec_city)
          return phrases
     

     # --- 30 ["root", 
     #              ["nsubj"], 
     #              ["obl", "case"], 
     #              ["conj", "case"]]
     def rule_9(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure
          for i in range(len(tokens)):
               is_rule_1 = [False, False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    cityHero = []
                    citySecHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              nameHero.append(tokens[property["index"]])
                              is_rule_1[0] = True

                         if property['rel'] == rels[2]:
                              cityHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[3]:
                                        cityHero.append(tokens[property_1["index"]])
                                        cityHero = sorted(cityHero, key=lambda x: x.start)
                                        is_rule_1[1] = True

                         if property['rel'] == rels[4]:
                              citySecHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[5]:
                                        citySecHero.append(tokens[property_1["index"]])
                                        citySecHero = sorted(citySecHero, key=lambda x: x.start)
                                        is_rule_1[2] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         name_sec_city = {"Name": nameHero, "City": citySecHero}
                         phrases.append(name_city)
                         phrases.append(name_sec_city)
          return phrases


     # --- 26 ["acl", 
     #              ["obl", 
     #                   ["case"], 
     #                   ["nsubj"]]]
     def rule_10(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure
          for i in range(len(tokens)):
               is_rule_1 = [False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              cityHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[2]:
                                        cityHero.append(tokens[property_1["index"]])
                                        cityHero = sorted(cityHero, key=lambda x: x.start)
                                        is_rule_1[0] = True
                                   if property_1['rel'] == rels[3]:
                                        nameHero.append(tokens[property_1["index"]])
                                        is_rule_1[1] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         phrases.append(name_city)
          return phrases


     # --- 29 ["obl", 
     #              ["case"], 
     #              ["nmod"], 
     #              ["nmod", "case"]]
     def rule_11(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure
          for i in range(len(tokens)):
               is_rule_1 = [False, False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              is_rule_1[0] = True

                         if property['rel'] == rels[2]:
                              nameHero.append(tokens[property["index"]])
                              is_rule_1[1] = True

                         if property['rel'] == rels[3]:
                              cityHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[4]:
                                        cityHero.append(tokens[property_1["index"]])
                                        cityHero = sorted(cityHero, key=lambda x: x.start)
                                        is_rule_1[2] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         phrases.append(name_city)
          return phrases


     # --- 31 ["root", 
     #              ["iobj"], 
     #              ["obl", "case"], 
     #              ["conj", "obl", "case"]
     def rule_12(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure
          for i in range(len(tokens)):
               is_rule_1 = [False, False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    cityHero = []
                    citySecHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              nameHero.append(tokens[property["index"]])
                              is_rule_1[0] = True

                         if property['rel'] == rels[2]:
                              cityHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[3]:
                                        cityHero.append(tokens[property_1["index"]])
                                        cityHero = sorted(cityHero, key=lambda x: x.start)
                                        is_rule_1[1] = True

                         if property['rel'] == rels[4]:
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[5]:
                                        citySecHero.append(tokens[property_1["index"]])
                                        for property_2 in connect_words[property_1["index"]]:
                                             if property_2['rel'] == rels[6]:
                                                  citySecHero.append(tokens[property_2["index"]])
                                                  citySecHero = sorted(citySecHero, key=lambda x: x.start)
                                                  is_rule_1[2] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         name_sec_city = {"Name": nameHero, "City": citySecHero}
                         phrases.append(name_city)
                         phrases.append(name_sec_city)
          return phrases


     # --- 32 ["root", 
     #              ["obl", "case"], 
     #              ["obl", "case", "conj"], 
     #              ["nsubj", "appos", "flat:name"]]
     def rule_13(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure
          for i in range(len(tokens)):
               is_rule_1 = [False, False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    nameFHero = []
                    nameSHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              cityHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[2]:
                                        cityHero.append(tokens[property_1["index"]])
                                        cityHero = sorted(cityHero, key=lambda x: x.start)
                                        is_rule_1[0] = True

                         if property['rel'] == rels[3]:
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[4]:
                                        nameFHero.append(tokens[property_1["index"]])
                                        for property_2 in connect_words[property_1["index"]]:
                                             if property_2['rel'] == rels[5]:
                                                  nameSHero.append(tokens[property_2["index"]])
                                                  is_rule_1[1] = True

                         if property['rel'] == rels[6]:
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[7]:
                                        nameHero.append(tokens[property_1["index"]])
                                        for property_2 in connect_words[property_1["index"]]:
                                             if property_2['rel'] == rels[8]:
                                                  nameHero.append(tokens[property_2["index"]])
                                                  nameHero = sorted(nameHero, key=lambda x: x.start)
                                                  is_rule_1[2] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         name_f_city = {"Name": nameFHero, "City": cityHero}
                         name_s_city = {"Name": nameSHero, "City": cityHero}
                         phrases.append(name_city)
                         phrases.append(name_f_city)
                         phrases.append(name_s_city)
          return phrases


     # --- 34 ["root", 
     #              ["nsubj"], 
     #              ["acl", "obl", "case"], 
     #              ["obl", "nmod"]]
     def rule_14(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure
          for i in range(len(tokens)):
               is_rule_1 = [False, False, False, False]
               if tokens[i].rel == rels[0]:
                    nameHero = []
                    nameFHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              nameHero.append(tokens[property["index"]])
                              is_rule_1[0] = True

                         if property['rel'] == rels[2]:
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[3]:
                                        cityHero.append(tokens[property_1["index"]])
                                        for property_2 in connect_words[property_1["index"]]:
                                             if property_2['rel'] == rels[4]:
                                                  cityHero.append(tokens[property_2["index"]])
                                                  cityHero = sorted(cityHero, key=lambda x: x.start)
                                                  is_rule_1[1] = True

                         if property['rel'] == rels[5]:
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[6]:
                                        nameFHero.append(tokens[property_1["index"]])
                                        is_rule_1[2] = True

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         name_f_city = {"Name": nameFHero, "City": cityHero}
                         phrases.append(name_city)
                         phrases.append(name_f_city)
          return phrases


     # ["root", ["nsubj", "appos", "conj"], 
     #         ["conj", "obl", 
     #                        ["amod"], 
     #                        ["case"]]]
     def rule_15(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure
          for i in range(len(tokens)):
               is_rule_1 = [False, False, False]

               if tokens[i].rel == rels[0]:
                    nameHero = []
                    nameFHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              nameHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[2]:
                                        nameHero.append(tokens[property_1["index"]])
                                        for property_2 in connect_words[property_1["index"]]:
                                             if property_2['rel'] == rels[3]:
                                                  nameFHero.append(tokens[property_2["index"]])
                                                  is_rule_1[0] = True

                         if property['rel'] == rels[4]:
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[5]:
                                        cityHero.append(tokens[property_1["index"]])
                                        for property_2 in connect_words[property_1["index"]]:
                                             if property_2['rel'] == rels[6]:
                                                  cityHero.append(tokens[property_2["index"]])
                                                  is_rule_1[1] = True

                                             if property_2['rel'] == rels[7]:
                                                  cityHero.append(tokens[property_2["index"]])
                                                  is_rule_1[2] = True
                         cityHero = sorted(cityHero, key=lambda x: x.start)

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         name_f_city = {"Name": nameFHero, "City": cityHero}
                         phrases.append(name_city)
                         phrases.append(name_f_city)
          return phrases
     
     # ["root", ["nsubj", "appos", "conj"], 
     #         ["obl", 
     #                ["amod"], 
     #                ["case"]]]
     def rule_16(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure
          for i in range(len(tokens)):
               is_rule_1 = [False, False, False]

               if tokens[i].rel == rels[0]:
                    nameHero = []
                    nameFHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              nameHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[2]:
                                        nameHero.append(tokens[property_1["index"]])
                                        for property_2 in connect_words[property_1["index"]]:
                                             if property_2['rel'] == rels[3]:
                                                  nameFHero.append(tokens[property_2["index"]])
                                                  is_rule_1[0] = True

                         if property['rel'] == rels[4]:
                              cityHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[5]:
                                        cityHero.append(tokens[property_1["index"]])
                                        is_rule_1[1] = True

                                   if property_1['rel'] == rels[6]:
                                        cityHero.append(tokens[property_1["index"]])
                                        is_rule_1[2] = True

                         cityHero = sorted(cityHero, key=lambda x: x.start)

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         name_f_city = {"Name": nameFHero, "City": cityHero}
                         phrases.append(name_city)
                         phrases.append(name_f_city)
          return phrases
     

     # ["root", ["nsubj"], 
     #         ["obl", 
     #                ["amod"], 
     #                ["case"]]]
     def rule_17(self, connect_words, *rels):
          phrases = []
          tokens = self.syntaxStructure
          for i in range(len(tokens)):
               is_rule_1 = [False, False, False]

               if tokens[i].rel == rels[0]:
                    nameHero = []
                    cityHero = []
                    for property in connect_words[i]:
                         if property['rel'] == rels[1]:
                              nameHero.append(tokens[property["index"]])
                              is_rule_1[0] = True

                         if property['rel'] == rels[2]:
                              cityHero.append(tokens[property["index"]])
                              for property_1 in connect_words[property["index"]]:
                                   if property_1['rel'] == rels[3]:
                                        cityHero.append(tokens[property_1["index"]])
                                        is_rule_1[1] = True

                                   if property_1['rel'] == rels[4]:
                                        cityHero.append(tokens[property_1["index"]])
                                        is_rule_1[2] = True

                         cityHero = sorted(cityHero, key=lambda x: x.start)

                    if all(is_rule_1):
                         name_city = {"Name": nameHero, "City": cityHero}
                         phrases.append(name_city)
          return phrases
     