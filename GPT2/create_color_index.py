import json


data=[]
color={'id':0, 'name':'red'}
data.append(color)
color={'id':1, 'name':'orange'}
data.append(color)
color={'id':2, 'name':'yellow'}
data.append(color)
color={'id':3, 'name':'green'}
data.append(color)
color={'id':4, 'name':'blue'}
data.append(color)
color={'id':5, 'name':'indigo'}
data.append(color)
color={'id':6, 'name':'violet'}
data.append(color)
color={'id':7, 'name':'purple'}
data.append(color)
color={'id':8, 'name':'magenta'}
data.append(color)
color={'id':9, 'name':'crimson'}
data.append(color)

color={'id':10, 'name':'scarlet'}
data.append(color)
color={'id':11, 'name':'vermilion'}
data.append(color)
color={'id':12, 'name':'ruby'}
data.append(color)
color={'id':13, 'name':'carmine'}
data.append(color)
color={'id':14, 'name':'cerise'}
data.append(color)
color={'id':15, 'name':'burgundy'}
data.append(color)
color={'id':16, 'name':'maroon'}
data.append(color)
color={'id':17, 'name':'auburn'}
data.append(color)
color={'id':18, 'name':'chestnut'}
data.append(color)
color={'id':19, 'name':'sienna'}
data.append(color)

color={'id':20, 'name':'amber'}
data.append(color)
color={'id':21, 'name':'saffron'}
data.append(color)
color={'id':22, 'name':'mustard'}
data.append(color)
color={'id':23, 'name':'cyan'}
data.append(color)
color={'id':24, 'name':'aquamarine'}
data.append(color)
color={'id':25, 'name':'teal'}
data.append(color)
color={'id':26, 'name':'jade'}
data.append(color)
color={'id':27, 'name':'chartreuse'}
data.append(color)
color={'id':28, 'name':'white'}
data.append(color)
color={'id':29, 'name':'black'}
data.append(color)

color={'id':30, 'name':'cerulean'}
data.append(color)
color={'id':31, 'name':'gray'}
data.append(color)
color={'id':32, 'name':'mauve'}
data.append(color)
color={'id':33, 'name':'fuchsia'}
data.append(color)
color={'id':34, 'name':'ochre'}
data.append(color)
color={'id':35, 'name':'khaki'}
data.append(color)
color={'id':36, 'name':'beige'}
data.append(color)
color={'id':37, 'name':'ivory'}
data.append(color)
color={'id':38, 'name':'ecru'}
data.append(color)
color={'id':39, 'name':'tan'}
data.append(color)

color={'id':40, 'name':'slate'}
data.append(color)
color={'id':41, 'name':'indigo'}
data.append(color)
color={'id':42, 'name':'sepia'}
data.append(color)
color={'id':43, 'name':'pink'}
data.append(color)
color={'id':44, 'name':'brown'}
data.append(color)
color={'id':45, 'name':'green'}
data.append(color)
color={'id':46, 'name':'color'}
data.append(color)
color={'id':47, 'name':'prune'}
data.append(color)
color={'id':48, 'name':'azure'}
data.append(color)
color={'id':49, 'name':'silver'}
data.append(color)
color={'id':50, 'name':'golden'}
data.append(color)
color={'id':50, 'name':'copper'}
data.append(color)




with open('data/MIdataset/color_index.json','w',encoding='utf-8')as f:
        json.dump(data,f,ensure_ascii=False,sort_keys=True)