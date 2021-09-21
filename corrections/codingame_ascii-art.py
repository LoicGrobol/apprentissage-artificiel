import sys
import math

# https://www.codingame.com/ide/puzzle/ascii-art

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ?"
letters = []
l = int(input())
h = int(input())
t = input()
t = t.upper()

res = ""
for i in range(h):
    row = input()
    for character in t:
        if character in alphabet:
            j = alphabet.index(character)
            res += "".join(row[j*l:j*l+l])
        else:
            res += "".join(row[-l:])
    res += "\n"
print(res)