import json


data=[]

sentiment={'id':1, 'name':'admiration', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':2, 'name':'amusement', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':3, 'name':'anger', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':4, 'name':'annoyance', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':5, 'name':'approval', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':6, 'name':'caring', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':7, 'name':'confusion', 'type': 'negative'}
data.append(sentiment)

sentiment={'id':10, 'name':'disappointment', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':11, 'name':'disapproval', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':12, 'name':'disgust', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':13, 'name':'embarrassment', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':15, 'name':'fear', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':16, 'name':'gratitude', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':17, 'name':'grief', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':18, 'name':'joy', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':19, 'name':'love', 'type': 'positive'}
data.append(sentiment)

sentiment={'id':20, 'name':'nervousness', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':21, 'name':'optimism', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':22, 'name':'pride', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':24, 'name':'relief', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':25, 'name':'remorse', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':26, 'name':'sadness', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':28, 'name':'anticipation', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':29, 'name':'pessimism', 'type': 'negative'}
data.append(sentiment)

sentiment={'id':30, 'name':'happy', 'type':'positive'}
data.append(sentiment)
sentiment={'id':31, 'name':'joyful', 'type':'positive'}
data.append(sentiment)
sentiment={'id':32, 'name':'excited', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':33, 'name':'enthusiastic', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':34, 'name':'optimistic', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':35, 'name':'grateful', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':36, 'name':'content', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':37, 'name':'proud', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':38, 'name':'inspired', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':39, 'name':'hopeful', 'type': 'positive'}
data.append(sentiment)

sentiment={'id':40, 'name':'loving', 'type':'positive'}
data.append(sentiment)
sentiment={'id':41, 'name':'passionate', 'type':'positive'}
data.append(sentiment)
sentiment={'id':42, 'name':'cheerful', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':43, 'name':'delighted', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':44, 'name':'elated', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':45, 'name':'delight', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':46, 'name':'cheer', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':47, 'name':'passion', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':48, 'name':'thrilled', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':49, 'name':'thrill', 'type': 'positive'}
data.append(sentiment)

sentiment={'id':50, 'name':'ecstatic', 'type':'positive'}
data.append(sentiment)
sentiment={'id':51, 'name':'blissful', 'type':'positive'}
data.append(sentiment)
sentiment={'id':52, 'name':'bliss', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':53, 'name':'satisfiled', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':54, 'name':'satisfy', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':55, 'name':'grate', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':56, 'name':'excit', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':57, 'name':'inspire', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':58, 'name':'hope', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':59, 'name':'peaceful', 'type': 'positive'}
data.append(sentiment)


sentiment={'id':61, 'name':'confidence', 'type':'positive'}
data.append(sentiment)
sentiment={'id':62, 'name':'courageous', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':63, 'name':'courage', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':64, 'name':'determined', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':65, 'name':'motivated', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':66, 'name':'energized', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':67, 'name':'refreshed', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':68, 'name':'relax', 'type': 'positive'}
data.append(sentiment)
sentiment={'id':69, 'name':'relaxed', 'type': 'positive'}
data.append(sentiment)

sentiment={'id':70, 'name':'calm', 'type':'positive'}
data.append(sentiment)
sentiment={'id':71, 'name':'serene', 'type':'positive'}
data.append(sentiment)
sentiment={'id':72, 'name':'sad', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':73, 'name':'angry', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':74, 'name':'frustrated', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':75, 'name':'disappointed', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':76, 'name':'anxious', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':77, 'name':'worry', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':78, 'name':'worried', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':79, 'name':'stress', 'type': 'negative'}
data.append(sentiment)

sentiment={'id':80, 'name':'stressful', 'type':'negative'}
data.append(sentiment)
sentiment={'id':81, 'name':'stressed', 'type':'negative'}
data.append(sentiment)
sentiment={'id':82, 'name':'depressed', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':83, 'name':'jealous', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':84, 'name':'envious', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':85, 'name':'envy', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':86, 'name':'resentful', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':87, 'name':'resent', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':88, 'name':'bitter', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':89, 'name':'guilty', 'type': 'negative'}
data.append(sentiment)

sentiment={'id':90, 'name':'ashamed', 'type':'negative'}
data.append(sentiment)
sentiment={'id':91, 'name':'ashame', 'type':'negative'}
data.append(sentiment)
sentiment={'id':92, 'name':'embarrassed', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':93, 'name':'lonely', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':94, 'name':'lone', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':95, 'name':'heartbroken', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':96, 'name':'miserable', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':97, 'name':'hopeless', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':98, 'name':'desperate', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':99, 'name':'irritated', 'type': 'negative'}
data.append(sentiment)

sentiment={'id':100, 'name':'annoy', 'type':'negative'}
data.append(sentiment)
sentiment={'id':101, 'name':'annoyed', 'type':'negative'}
data.append(sentiment)
sentiment={'id':102, 'name':'disgusted', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':103, 'name':'offend', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':104, 'name':'offended', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':105, 'name':'humiliated', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':106, 'name':'insecure', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':107, 'name':'vulmerable', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':108, 'name':'overwhelmed', 'type': 'negative'}
data.append(sentiment)
sentiment={'id':109, 'name':'exhausted', 'type': 'negative'}
data.append(sentiment)

with open('data/MIdataset/sentiment_index.json','w',encoding='utf-8')as f:
        json.dump(data,f,ensure_ascii=False,sort_keys=True)