import json


data=[]
country={'id':1, 'name':'Russia', 'land_area':17098246, 'time':1991}
data.append(country)
country={'id':2, 'name':'Canada', 'land_area':9984670, 'time':1867}
data.append(country)
country={'id':3, 'name':'United States', 'land_area':9833520, 'time':1776}
data.append(country)
country={'id':4, 'name':'China', 'land_area':9596960, 'time':1949}
data.append(country)
country={'id':5, 'name':'Brazil', 'land_area':8515770, 'time':1822}
data.append(country)
country={'id':6, 'name':'Australia', 'land_area':7692024, 'time':1901}
data.append(country)
country={'id':7, 'name':'India', 'land_area':3287263, 'time':1947}
data.append(country)
# country={'id':8, 'name':'Argentina', 'land_area':2780400, 'time':1816}
# data.append(country)
# country={'id':9, 'name':'Kazakhstan', 'land_area':2724900, 'time':1991}
# data.append(country)
# country={'id':10, 'name':'Algeria', 'land_area':2381741, 'time':1962}
# data.append(country)

country={'id':11, 'name':'Congo', 'land_area':2344858, 'time':1960}
data.append(country)
country={'id':12, 'name':'Saudi Arabia', 'land_area':2149690, 'time':1932}
data.append(country)
country={'id':13, 'name':'Mexico', 'land_area':1964375, 'time':1810}
data.append(country)
# country={'id':14, 'name':'Indonesia', 'land_area':1904569, 'time':1945}
# data.append(country)
# country={'id':15, 'name':'Sudan', 'land_area':1861484, 'time':1956}
# data.append(country)
country={'id':16, 'name':'Libya', 'land_area':1759540, 'time':1951}
data.append(country)
country={'id':17, 'name':'Iran', 'land_area':1648195, 'time':1979}
data.append(country)
# country={'id':18, 'name':'Mongolia', 'land_area':1564110, 'time':1921}
# data.append(country)
country={'id':19, 'name':'Peru', 'land_area':1285216, 'time':1821}
data.append(country)
country={'id':20, 'name':'Chad', 'land_area':1284000, 'time':1960}
data.append(country)

country={'id':21, 'name':'Niger', 'land_area':1267000, 'time':1960}
data.append(country)
country={'id':22, 'name':'Angola', 'land_area':1246700, 'time':1975}
data.append(country)
country={'id':23, 'name':'Mali', 'land_area':1240192, 'time':1960}
data.append(country)
country={'id':24, 'name':'South Africa', 'land_area':1219090, 'time':1910}
data.append(country)
# country={'id':25, 'name':'Colombia', 'land_area':1141748, 'time':1810}
# data.append(country)
# country={'id':26, 'name':'Ethiopia', 'land_area':1104300, 'time':1855}
# data.append(country)
# country={'id':27, 'name':'Bolivia', 'land_area':1098581, 'time':1825}
# data.append(country)
# country={'id':28, 'name':'Mauritania', 'land_area':1030700, 'time':1960}
# data.append(country)
country={'id':29, 'name':'Egypt', 'land_area':1001450, 'time':1922}
data.append(country)
# country={'id':30, 'name':'Tanzania', 'land_area':947300, 'time':1961}
# data.append(country)

# country={'id':31, 'name':'Nigeria', 'land_area':923768, 'time':1960}
# data.append(country)
# country={'id':32, 'name':'Venezuela', 'land_area':912050, 'time':1811}
# data.append(country)
country={'id':33, 'name':'Pakistan', 'land_area':881912, 'time':1947}
data.append(country)
# country={'id':34, 'name':'Namibia', 'land_area':825615, 'time':1990}
# data.append(country)
# country={'id':35, 'name':'Mozambique', 'land_area':786380, 'time':1975}
# data.append(country)
country={'id':36, 'name':'Turkey', 'land_area':783562, 'time':1923}
data.append(country)
country={'id':37, 'name':'Chile', 'land_area':756102, 'time':1810}
data.append(country)
# country={'id':38, 'name':'Zambia', 'land_area':752618, 'time':1964}
# data.append(country)
country={'id':39, 'name':'Myanmar', 'land_area':676578, 'time':1948}
data.append(country)
# country={'id':40, 'name':'Afghanistan', 'land_area':652230, 'time':1919}
# data.append(country)

country={'id':41, 'name':'South Sudan', 'land_area':644329, 'time':2011}
data.append(country)
country={'id':42, 'name':'France', 'land_area':643801, 'time':843}
data.append(country)
# country={'id':43, 'name':'Somalia', 'land_area':637657, 'time':1960}
# data.append(country)
# country={'id':44, 'name':'Central African Republic', 'land_area':622984, 'time':1960}
# data.append(country)
country={'id':45, 'name':'Ukraine', 'land_area':603550, 'time':1991}
data.append(country)
country={'id':46, 'name':'Madagascar', 'land_area':587041, 'time':1960}
data.append(country)
# country={'id':47, 'name':'Botswana', 'land_area':581730, 'time':1966}
# data.append(country)
country={'id':48, 'name':'Kenya', 'land_area':580367, 'time':1963}
data.append(country)
country={'id':49, 'name':'Yemen', 'land_area':527968, 'time':1990}
data.append(country)
country={'id':50, 'name':'Thailand', 'land_area':513120, 'time':1238}
data.append(country)

country={'id':51, 'name':'Spain', 'land_area':505990, 'time':1469}
data.append(country)
# country={'id':52, 'name':'Turkmenistan', 'land_area':488100, 'time':1991}
# data.append(country)
# country={'id':53, 'name':'Cameroon', 'land_area':475440, 'time':1960}
# data.append(country)
# country={'id':54, 'name':'Papua New Guinea', 'land_area':462840, 'time':1975}
# data.append(country)
country={'id':55, 'name':'Sweden', 'land_area':450295, 'time':1523}
data.append(country)
# country={'id':56, 'name':'Uzbekistan', 'land_area':447400, 'time':1991}
# data.append(country)
# country={'id':57, 'name':'Morocco', 'land_area':446550, 'time':1956}
# data.append(country)
country={'id':58, 'name':'Iraq', 'land_area':438317, 'time':1932}
data.append(country)
# country={'id':59, 'name':'Paraguay', 'land_area':406752, 'time':1811}
# data.append(country)
country={'id':60, 'name':'Zimbabwe', 'land_area':390757, 'time':1980}
data.append(country)

country={'id':61, 'name':'Japan', 'land_area':377930, 'time':660}
data.append(country)
country={'id':62, 'name':'Germany', 'land_area':357114, 'time':1871}
data.append(country)
country={'id':63, 'name':'Singapore', 'land_area':581, 'time':1965}
data.append(country)
country={'id':64, 'name':'Denmark', 'land_area':2166086, 'time':1849}
data.append(country)
# country={'id':65, 'name':'Vietnam', 'land_area':313429, 'time':1945}
# data.append(country)
# country={'id':66, 'name':'Malaysia', 'land_area':328657, 'time':1963}
# data.append(country)
country={'id':67, 'name':'Norway', 'land_area':304282, 'time':1814}
data.append(country)
country={'id':68, 'name':'Poland', 'land_area':304255, 'time':1989}
data.append(country)
country={'id':69, 'name':'Italy', 'land_area':295717, 'time':1946}
data.append(country)
country={'id':70, 'name':'Philippines', 'land_area':298170, 'time':1899}
data.append(country)

country={'id':71, 'name':'Gabon', 'land_area':257670, 'time':1960}
data.append(country)
country={'id':72, 'name':'United Kingdom', 'land_area':242741, 'time':1922}
data.append(country)
# country={'id':73, 'name':'Uganda', 'land_area':227533, 'time':1962}
# data.append(country)
country={'id':74, 'name':'Romania', 'land_area':230080, 'time':1859}
data.append(country)
country={'id':75, 'name':'Belarus', 'land_area':202900, 'time':1991}
data.append(country)
country={'id':76, 'name':'Syria', 'land_area':183630, 'time':1946}
data.append(country)
# country={'id':77, 'name':'Uruguay', 'land_area':175015, 'time':1825}
# data.append(country)
# country={'id':78, 'name':'Greece', 'land_area':128900, 'time':1830}
# data.append(country)
country={'id':79, 'name':'Cuba', 'land_area':103800, 'time':1902}
data.append(country)
country={'id':80, 'name':'Iceland', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':81, 'name':'Austria', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':82, 'name':'Ireland', 'land_area':103800, 'time':1902}
data.append(country)
country={'id':83, 'name':'Georgia', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':84, 'name':'Latvia', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':85, 'name':'Slovakia', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':86, 'name':'Estonia', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':87, 'name':'Switzerland', 'land_area':103800, 'time':1902}
data.append(country)
country={'id':88, 'name':'Bhutan', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':89, 'name':'Belgium', 'land_area':100830, 'time':1944}
data.append(country)

country={'id':90, 'name':'Haiti', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':91, 'name':'Israel', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':92, 'name':'Bahamas', 'land_area':103800, 'time':1902}
data.append(country)
country={'id':93, 'name':'Vanuatu', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':94, 'name':'Gambia', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':95, 'name':'Jamaica', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':96, 'name':'Kosovo', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':97, 'name':'Cyprus', 'land_area':103800, 'time':1902}
data.append(country)
country={'id':98, 'name':'Lebanon', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':99, 'name':'Abkhazia', 'land_area':100830, 'time':1944}
data.append(country)

country={'id':100, 'name':'Samoa', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':101, 'name':'Mauritius', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':102, 'name':'Comoros', 'land_area':103800, 'time':1902}
data.append(country)
country={'id':103, 'name':'Finland', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':104, 'name':'Kiribati', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':105, 'name':'Bahrain', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':106, 'name':'Dominica', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':107, 'name':'Tonga', 'land_area':103800, 'time':1902}
data.append(country)
country={'id':108, 'name':'Andorra', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':109, 'name':'Palau', 'land_area':100830, 'time':1944}
data.append(country)

country={'id':110, 'name':'Ecuador', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':111, 'name':'Ghana', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':112, 'name':'Laos', 'land_area':103800, 'time':1902}
data.append(country)
country={'id':113, 'name':'Guyana', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':114, 'name':'Kyrgyzstan', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':115, 'name':'Senegal', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':116, 'name':'Cambodia', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':117, 'name':'Nepal', 'land_area':103800, 'time':1902}
data.append(country)
country={'id':118, 'name':'Suriname', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':119, 'name':'Tunisia', 'land_area':100830, 'time':1944}
data.append(country)

country={'id':120, 'name':'Paris', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':121, 'name':'London', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':122, 'name':'Rome', 'land_area':103800, 'time':1902}
data.append(country)
country={'id':123, 'name':'Berlin', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':124, 'name':'Barcelona', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':125, 'name':'Sydney', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':126, 'name':'Tokyo', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':127, 'name':'Beijing', 'land_area':103800, 'time':1902}
data.append(country)
country={'id':128, 'name':'Mumbai', 'land_area':100830, 'time':1944}
data.append(country)
country={'id':129, 'name':'Chicago', 'land_area':100830, 'time':1944}
data.append(country)

with open('data/MIdataset/index.json','w',encoding='utf-8')as f:
        json.dump(data,f,ensure_ascii=False,sort_keys=True)