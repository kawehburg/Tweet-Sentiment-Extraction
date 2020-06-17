
def map_offset(tweet, offsets, raw_tweet):
    raw_tweet = raw_tweet + ' '
    j = 0
    s = []
    last_space = 0
    for i in range(len(raw_tweet)):
        if raw_tweet[i] == ' ':
            last_space = i
            if len(s) > 0:
                s[-1][1] = i
        if j == len(tweet):
            continue
        if raw_tweet[i] == tweet[j]:
            if j > 0 and tweet[j - 1] == ' ':
                s.append([last_space + 1, i + 1])
            else:
                s.append([i, i + 1])
            j += 1
    for i in range(len(s) - 1):
        s[i][1] = s[i + 1][0]
        
    raw_offset = []
    for item in offsets:
        raw_offset.append([s[item[0]][0], s[item[1] - 1][1]])

    return raw_offset
    

tweet = " shared kim hiltermand - portfolio: shared by kaare finally a dane iâ've got the honor to do the amazing.. http://tinyurl.com/coypsl"
raw_tweet = " shared kim hi¿½ltermand - portfolio: shared by kaare finally a dane iâ've got the honor to do the amazing.. http://tinyurl.com/coypsl"
offsets = [[0, 1], [2, 6], [7, 8], [8, 10]]
raw_offset = map_offset(tweet, offsets, raw_tweet)
print(raw_offset)
for item in offsets:
    print(tweet[item[0]: item[1]], end='-')
print()
for item in raw_offset:
    print(raw_tweet[item[0]: item[1]], end='-')
print()

