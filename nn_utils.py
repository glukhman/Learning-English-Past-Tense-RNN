import random, math

def read_verbs():
    with open("most-common-verbs-english.csv", "r") as f:
        raw_data = f.readlines()
    verb_list = []
    for entry in raw_data:
        entry = entry.replace('\n', '').replace(' ', '').split(',')
        v1, v2 = entry[0], entry[3]
        verb_list.append((v1, v2))
    return verb_list[1:]    # exclude title

"""
IPA transcriptions based on the CMU Pronouncing Dictionary, created by the
Speech Group at Carnegie Mellon University
(https://en.wikipedia.org/wiki/CMU_Pronouncing_Dictionary)
BSD license.       
Transcribed in IPA by Noah Constant, UMass Amherst
(https://people.umass.edu/nconstan/CMU-IPA/)
"""
def read_ipa_transcripts():
    with open("CMU.in.IPA.txt", "r", encoding="utf-8") as f:       
        raw_data = f.readlines()
    spelling_to_IPA = {}
    for entry in raw_data:            
        entry = entry.replace('\n', '').replace('\t', '').replace('ˈ', '')\
                .replace(' ', '').replace('ˌ', '').split(',')
        if '(' in entry[0]:
            continue            # ignore pronunciation variations
        spelling = entry[0]
        IPA = entry[1].replace('aj', 'Y').replace('ej', 'E')\
              .replace('ow', 'O').replace('aw', 'W')\
              .replace('ɔj', 'Ø').replace('ɚ', 'ɜɹ')
              # represent diphthongs with a single symbol
              # and avoid complications with [ɚ] vs. [ʌɹ]
        spelling_to_IPA[spelling] = IPA
    return spelling_to_IPA

def transcribe_verbs(verbs, IPA_glossary):
    transcribed = []
    for v1, v2 in verbs:
        regular = v2.endswith('ed') and v2 not in ['sped','bred','fed','fled',
                                                   'bled', 'led','shed','wed']
        try:
            v1, v2 = IPA_glossary[v1], IPA_glossary[v2]
        except KeyError:
            continue    # missing transcription for verb
        # the following is only for unified transcription convention:
        if regular and v2.endswith('ʌd'):
            v2 = v2[:-2]+'ɪd'
        transcribed.append((v1, v2, regular))
    return transcribed

phonemes = ['p','b','t','d','ʈ','ɖ','c','ɟ','k','g','q','ɢ','ʔ','ʦ','ʣ','ʧ',
            'ʤ','ɸ','β','f','v','θ','ð','s','z','ɬ','ɮ','ʃ','ʒ','ʂ','ʐ','ɕ',
            'ʑ','ç','ʝ','ɧ','x','ɣ','χ','ʁ','ħ','ʕ','h','ɦ','m','ɱ','n','ɳ',
            'ɲ','ŋ','ɴ','ʙ','ⱱ','ɾ','r','ɺ','ɽ','ʀ','ʜ','ʢ','ʋ','ɹ','l','ɻ',
            'ɭ','j','ɥ','ʎ','ɰ','ʍ','w','ʟ','i','y','I','ɨ','ʉ','ɯ','u','U',
            'ɪ','ʏ','ʊ','e','ø','E','ɘ','ɵ','ɤ','o','O','ɛ','œ','ɜ','ɞ','ə',
            'Ø','ʌ','ɔ','æ','ɐ','a','ɶ','Y','W','ɑ','ɒ']

def sonority(phoneme):
    if phoneme in ['p','b','t','d','ʈ','ɖ','c','ɟ','k','g','q','ɢ','ʔ']:
        return 0
    elif phoneme in ['ʦ','ʣ','ʧ','ʤ']: return 1
    elif phoneme in ['ɸ','β','f','v','θ','ð','s','z','ɬ','ɮ','ʃ','ʒ','ʂ','ʐ',
                     'ɕ','ʑ','ç','ʝ','ɧ','x','ɣ','χ','ʁ','ħ','ʕ','h','ɦ']:
        return 2
    elif phoneme in ['m','ɱ','n','ɳ','ɲ','ŋ','ɴ']: return 3
    elif phoneme in ['ʙ','ⱱ','ɾ','r','ɺ','ɽ','ʀ','ʜ','ʢ']: return 4
    elif phoneme in ['ʋ','ɹ','l','ɻ','ɭ','j','ɥ','ʎ','ɰ','ʍ','w','ʟ']:
                     return 5
    elif phoneme in ['i','y','I','ɨ','ʉ','ɯ','u','U']: return 6
    elif phoneme in ['ɪ','ʏ','ʊ']: return 7
    elif phoneme in ['e','ø','E','ɘ','ɵ','ɤ','o','O']: return 8
    elif phoneme in ['ə','Ø']: return 9
    elif phoneme in ['ɛ','œ','ɜ','ɞ','ʌ','ɔ']: return 10
    elif phoneme in ['æ','ɐ']: return 11
    elif phoneme in ['a','ɶ','Y','W','ɑ','ɒ']: return 12
    else: return -10

def backness(phoneme):
    if phoneme in ['p','b','ɸ','β','m','ʙ']: return 0
    elif phoneme in ['f','v','ɱ','ⱱ','ʋ']: return 1
    elif phoneme in ['θ','ð']: return 2
    elif phoneme in ['t','d','ʦ','ʣ','s','z','ɬ','ɮ','n','ɾ','r','ɺ','ɹ','l']:
        return 3
    elif phoneme in ['ʈ','ɖ','ʧ','ʤ','ʃ','ʒ','ʂ','ʐ','ɕ','ʑ','ɳ','ɽ','ɻ','ɭ']:
        return 4
    elif phoneme in ['c','ɟ','ç','ʝ','ɲ','j','ɥ','ʎ','i','y','I','ɪ','ʏ']:
        return 5
    elif phoneme in ['ɧ','ɨ','ʉ','e','ø','E']: return 6
    elif phoneme in ['k','g','x','ɣ','ŋ','ɰ','ʍ','w','ʟ','ɯ','u','U','ʊ','ɘ',
                     'ɵ','ɛ','œ']: return 7
    elif phoneme in ['q','ɢ','χ','ʁ','ɴ','ʀ','ɤ','o','O','ɜ','ɞ','ə','Ø','æ',
                     'a','ɶ']: return 8
    elif phoneme in ['ħ','ʕ','ʜ','ʢ','ʌ','ɔ','ɐ','Y','W']: return 9
    elif phoneme in ['ʔ','h','ɦ','ɑ','ɒ']: return 10
    else: return -10

def VOT(phoneme):
    if phoneme in ['p','t','ʈ','c','k','q','ʔ','ʦ','ʧ','ɸ','f','θ','s','ɬ','ʃ',
                   'ʂ','ɕ','ç','ɧ','x','χ','ħ','h','ʜ','ʍ']: return 1
    else: return 0

def rounded(phoneme):
    if phoneme in ['ʃ','ʒ','ɹ','ɻ','ɥ','ʍ','w','y','ʉ','u','U','ʏ','ʊ','ø','ɵ',
                   'o','O','œ','ɞ','Ø','ɔ','ɶ','W','ɒ']: return 1
    else: return 0

def palatalized(phoneme):
    if phoneme in ['ɕ','ʑ']: return 1
    else: return 0

def lateral(phoneme):
    if phoneme in ['ɬ','ɮ','ɺ','l','ɭ','ʎ','ʟ']: return 1
    else: return 0

def nasal(phoneme):
    if phoneme in ['m','ɱ','n','ɳ','ɲ','ŋ','ɴ']: return 1
    else: return 0

def sibilant(phoneme):
    if phoneme in ['ʦ','ʣ','ʧ','ʤ','s','z','ʃ','ʒ','ʂ','ʐ','ɕ','ʑ']: return 1
    else: return 0

def trilled(phoneme):
    if phoneme in ['ʙ','r','ʀ','ʜ','ʢ']: return 1
    else: return 0

def diphthong(phoneme):
    if phoneme in ['I','U','E','O','Ø','Y','W']: return 1
    else: return 0