import os
import re
from pythainlp import word_tokenize

# Ensure UTF-8 encoding is set
os.environ['PYTHONIOENCODING'] = 'utf-8'

def english_to_thai_fallback(word):
    """
    Simple manual fallback to convert English words to Thai phonemes.
    This is a placeholder and should be replaced with a proper library if needed.
    """
    mapping = {
        "today": "ทูเด",
        "hello": "เฮลโล",
        "world": "เวิลด์",
        "computer": "คอมพิวเตอร์",
        "phone": "โฟน",
        "school": "สคูล",
        "teacher": "ทีเชอร์",
        "student": "สตูเดนท์",
        "apple": "แอปเปิล",
        "orange": "ออเรนจ์",
        "table": "เทเบิล",
        "chair": "แชร์",
        "window": "วินโดว์",
        "door": "ดอร์",
        "water": "วอเทอร์",
        "coffee": "คอฟฟี่",
        "milk": "มิลค์",
        "juice": "จูซ",
        "food": "ฟูด",
        "car": "คาร์",
        "bus": "บัส",
        "train": "เทรน",
        "airplane": "แอร์เพลน",
        "boat": "โบ๊ท",
        "dog": "ด็อก",
        "cat": "แคท",
        "bird": "เบิร์ด",
        "fish": "ฟิช",
        "house": "เฮ้าส์",
        "city": "ซิตี้",
        "country": "คันทรี",
        "family": "แฟมิลี",
        "friend": "เฟรนด์",
        "love": "เลิฟ",
        "happiness": "แฮปปิเนส",
        "sadness": "แซดเนส",
        "anger": "แองเกอร์",
        "smile": "สไมล์",
        "cry": "คราย",
        "laugh": "ลาฟ",
        "light": "ไลท์",
        "dark": "ดาร์ก",
        "sun": "ซัน",
        "moon": "มูน",
        "star": "สตาร์",
        "ocean": "โอเชียน",
        "mountain": "เมาเทน",
        "river": "ริเวอร์",
        "forest": "ฟอเรสต์",
        "i": "ไอ",
        "love": "เลิฟ",
        "you": "ยู",
        "talk": "ทอล์ก",
        "sing": "ซิง",
        "dance": "แดนซ์",
        "read": "รีด",
        "write": "ไรท์",
        "run": "รัน",
        "walk": "วอล์ค",
        "jump": "จัมป์",
        "swim": "สวิม",
        "eat": "อีท",
        "drink": "ดริงค์",
        "sleep": "สลีป",
        "wake": "เวค",
        "good": "กู๊ด",
        "bad": "แบด",
        "happy": "แฮปปี้",
        "sad": "แซด",
        "angry": "แองกรี",
        "tired": "ไทร์ด"
        # Add more mappings as needed
    }
    # Add mappings for individual English characters
    character_mapping = {
        "a": "เอ",
        "b": "บี",
        "c": "ซี",
        "d": "ดี",
        "e": "อี",
        "f": "เอฟ",
        "g": "จี",
        "h": "เอช",
        "i": "ไอ",
        "j": "เจ",
        "k": "เค",
        "l": "แอล",
        "m": "เอ็ม",
        "n": "เอ็น",
        "o": "โอ",
        "p": "พี",
        "q": "คิว",
        "r": "อาร์",
        "s": "เอส",
        "t": "ที",
        "u": "ยู",
        "v": "วี",
        "w": "ดับเบิลยู",
        "x": "เอ็กซ์",
        "y": "วาย",
        "z": "แซด"
    }
    number_mapping = {
        "0": "ศูนย์",
        "1": "หนึ่ง",
        "2": "สอง",
        "3": "สาม",
        "4": "สี่",
        "5": "ห้า",
        "6": "หก",
        "7": "เจ็ด",
        "8": "แปด",
        "9": "เก้า",
        "10": "สิบ",
        "20": "ยี่สิบ",
        "30": "สามสิบ",
        "40": "สี่สิบ",
        "50": "ห้าสิบ",
        "60": "หกสิบ",
        "70": "เจ็ดสิบ",
        "80": "แปดสิบ",
        "90": "เก้าสิบ",
        "100": "หนึ่งร้อย"
    }
    mapping.update(number_mapping)
    mapping.update(character_mapping)
    return mapping.get(word.lower(), word)

def clean_thai_text(text):
    # Function to duplicate words followed by "ๆ"
    def duplicate_word(match):
        word = match.group(1)
        return word + word

    # Replace "ๆ" with duplicated word
    #text = re.sub(r'(\S+?)ๆ', duplicate_word, text)

    # Tokenize the text
    words = word_tokenize(text, keep_whitespace=True)

    # Convert English words to Thai phonemes
    cleaned_text = []
    for word in words:
        if re.search(r'[a-zA-Z]', word):  # If the word contains English letters
            try:
                from pythainlp import transliterate  # Import here to handle the library conditionally
                thai_phoneme = transliterate(word, engine='ipa')
                cleaned_text.append(thai_phoneme)
            except Exception as e:
                #print(f"Error transliterating word '{word}': {e}")
                # Use fallback for transliteration
                cleaned_text.append(english_to_thai_fallback(word))
        else:
            cleaned_text.append(word)

    return ''.join(cleaned_text)
