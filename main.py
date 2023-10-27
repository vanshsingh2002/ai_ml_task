import requests
from translate import Translator
import googletrans
from langdetect import detect
import pandas as pd
import recommendation


def create_hinglish_translation(english_text):
    words = english_text.split()
    hinglish_translation = []
    
    for word in words:
        detected_lang = detect(word)
      
        if detected_lang != 'en':
            translator = Translator(to_lang="hi")
            translated_word = translator.translate(word)
            hinglish_translation.append(translated_word)
        else:
            hinglish_translation.append(word)

    return ' '.join(hinglish_translation)

def hindi_to_english(text):
    translator = Translator(to_lang="en")
    translation = translator.translate(text)
    return translation

def translate_hindi_to_english(hindi_word):
  translator = googletrans.Translator()
  translation = translator.translate(hindi_word, src='hi', dest='en')
  return translation.text

def translate_to_hindi(text):
    translator = Translator(to_lang="hi")
    translation = translator.translate(text)
    return translation


if __name__ == "__main__":
    english_statements = [
        "I was waiting for my bag"
    ]
    
    train_x_data = pd.read_csv('static/train_x.csv')['restaurant'].tolist()
    train_y_data = pd.read_csv('static/train_y.csv')['restaurant'].tolist()
    
    rec_sys = recommendation.AccessoriesRecommendation(train_x_data, train_y_data)

    whole = []
    for statement in english_statements:
        hinglish_translation = translate_to_hindi(statement)
        hinglish_translation_list = hinglish_translation.split(" ")
        res = ""
        for item in hinglish_translation_list:
            hindi_item = translate_hindi_to_english(item)
            recommendations = rec_sys.recommend_phone(hindi_item)
            if recommendations:
                res += recommendations[0] + " "
            else:
                res += item +" "
        whole.append(res)
                
    print(whole)   