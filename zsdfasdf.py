from konlpy.tag import Mecab
tokenizer = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
print(tokenizer.morphs("이근범"))
