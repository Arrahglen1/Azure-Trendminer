#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Library to do API calls
import requests
import json
import random


# In[ ]:


#The code gives just the jokes for twopart, setup and delivery
setup =""
delivery =" "
params = {'category': 'Programming', 'type': 'twopart', 'flags': {'nsfw': False, 'religious': False, 'political': False, 'racist': False, 'sexist': False}}
url = 'https://sv443.net/jokeapi/v2/joke/Programming'
result1 = requests.get(url, params=params)
result2 = result1.json()
for key, vals in result2.items():
    if key =="setup":
        setup = vals
    if key == "delivery":
        delivery = vals
mydictionarry = {
    "category": "Programming",
    "type": "twopart",
    "setup": setup,
    "delivery": delivery,
    "flags": {
        "nsfw": "false",
        "religious": "false",
        "political": "false",
        "racist": "false",
        "sexist": "false"
    }
}
for keys, val in mydictionarry.items():
    if val == "twopart":
        if keys == "setup":
            val = setup
        if keys == "delivery":
            val = delivery
    if keys =="setup" :
        print (" {}: {}".format(keys, val))
    if keys =="delivery":
        print (" {} :{}".format(keys, val))



# In[ ]:


#The code gives just the jokes for the single part
joke =""
params1 = {'category': 'Programming', 'type': 'single', 'flags': {'nsfw': False, 'religious': False, 'political': False, 'racist': False, 'sexist': False}}
url2 = 'https://sv443.net/jokeapi/v2/joke/Programming'
sult = requests.get(url2, params=params1)
sult2 = sult.json()
for key, vals in sult2.items():
    if key =="joke":
        joke = vals    
mydictionarry = {
    "category": "Programming",
    "type": "single",
    "joke": joke,
    "flags": {
        "nsfw": "false",
        "religious": "false",
        "political": "false",
        "racist": "false",
        "sexist": "false"
    }
}
for keys, val in mydictionarry.items():
    if val == "single":
        if keys == "joke":
            val = joke
    if keys =="joke" :
        print (" {}: {}".format(keys, val))
    



# In[ ]:


#using a different url to get 10 random jokes
url2 = 'http://api.icndb.com/jokes/random/10'
sult = requests.get(url2)
print(sult.text)
sult1 = requests.get(url2)
sult2 = sult1.json()


# In[ ]:
#'ID range': {'idRange': [1-2,4-5,6-8,9-11]},
   # "ID range": {
   # "idRange": 1-2,
   # "idRange": 4-5,
   # "idRange": 6-8,
   ## "idRange": 9-11
   # },



