#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Library to do API calls
import requests
import json
import random


# In[ ]:


# An API call can have different types (GET / POST / PUT / DELETE)
# Example of a simple GET to get a random joke
payload = {'category': 'Programming', 'type': 'twopart', 'flags': {'nsfw': False, 'religious': False, 'political': False, 'racist': False, 'sexist': False}}
payloads =  {'category': 'Programming', 'type': 'single', 'flags': {'nsfw': True, 'religious': True, 'political': False, 'racist': False, 'sexist': False}}
result = requests.get('https://sv443.net/jokeapi/v2/joke/Programming?type=twopart' , params=payload)
ralo = requests.get('https://sv443.net/jokeapi/v2/joke/Programming,Dark?type=twopart', params=payloads)
trado = requests.get('https://sv443.net/jokeapi/v2/joke/Programming?blacklistFlags=nsfw,religious,political&type=twopart&idRange=216')
###########
# If you play around with the "Try it out here:" on https://sv443.net/jokeapi/v2/#getting-started
# you should be able to write a REST call where you only get jokes about (1) programming, (2) are in two parts
# (3) containt the string "java"
# Please do so as an exercise! Look at the URL changes in the example at https://sv443.net/jokeapi/v2/#getting-started


# In[ ]:


#check the status code of the ralo,trado and result
ralo.status_code
trado.status_code
result.status_code


# In[ ]:


#check if the status code is ok
ralo.ok
trado.ok
result.ok


# In[ ]:


# The result object contains text which a JSON representation of the data you received through the Joke API

#print(ralo.json())
#print(ralo.url)
#print(trado.json())
#print(trado.url)
print(result.json())
print(result.url)


# In[ ]:


#specifying the category,format,blacklist flags,joke type and id range
response = requests.get('https://sv443.net/jokeapi/v2/joke/Programming?blacklistFlags=nsfw,religious,political&type=twopart&Category=Programming&idRange=13-16')


# In[ ]:


#convert the JSON file into a Python dictionary
print(response.json())


# In[ ]:


#parameters which you can include in your get request to send more data about a particular request
response = requests.get('https://sv443.net/jokeapi/v2/joke/Programming?blacklistFlags=racist,sexist&type=twopart', 
                        headers={"Accept": "application/json"}, params={"key1": "value1"})


# In[ ]:


print(response.json())


# In[ ]:


# The result object contains text which a JSON representation of the data you received through the API
jcon = trado.text
print(jcon)


# In[ ]:


# The result object contains text which a JSON representation of the data you received through the API
jsdata = result.text
print(jsdata)


# In[ ]:


jsdatas = ralo.text
print(jsdatas)


# In[ ]:


# You can navigate the json object and get the information you're interested in
# In case it is a simple joke there is a key called "joke"
jsdict = json.loads(jsdata)
if "joke" in jsdict:
    print("Joke: " + jsdict["joke"])
else:
    print("Setup: " + jsdict["setup"])
    print("Delivery: " + jsdict["delivery"])


# In[ ]:


# You can pass parameters in a REST call to only get particular types of jokes
# This can be done by updating the url
# In this case you'll only get jokes without setup / delivery
result2 = requests.get('https://sv443.net/jokeapi/v2/joke/Any?type=single')
jsdata2 = result2.text
jsdict2 = json.loads(jsdata2)
print("Joke: " + jsdict2["joke"])


# In[ ]:


# Alternatively (and better) you can pass parameters in the requests call
params = {'type': 'single'}
result3 = requests.get('https://sv443.net/jokeapi/v2/joke/Any', params=params)
jsdata3 = result3.text
jsdict3 = json.loads(jsdata3)
print("Joke: " + jsdict3["joke"])


# In[ ]:


###########
# If you play around with the "Try it out here:" on https://sv443.net/jokeapi/v2/#getting-started
# you should be able to write a REST call where you only get jokes about (1) programming, (2) are in two parts
# (3) containt the string "java"
# Please do so as an exercise! Look at the URL changes in the example at https://sv443.net/jokeapi/v2/#getting-started


# In[ ]:


#########
# Up until here all we've done is using GET requests
# Next exercise is actually submitting a new joke by using a PUT request
# This will need to be done on the following url https://sv443.net/jokeapi/v2/submit
# You'll need to send the payload as a json
# You can find the right format of the json at https://sv443.net/jokeapi/v2/#getting-started under submit joke
# You'll have to generate a json object for this in python (use the json package imported above)
# You'll have to use requests.put


# In[ ]:


Dalo = requests.put('https://sv443.net/jokeapi/v2/submit')


# In[ ]:


#check the status code of the Dalo
Dalo.status_code


# In[ ]:


print(Dalo.json())


# In[ ]:

#payload = {'key1': 'value1', 'key2': 'value2'}

#r = requests.post("https://httpbin.org/post", data=payload)
#print(r.text)


# In[ ]:


https://requests.readthedocs.io/en/master/user/quickstart/


# In[ ]:
#Using postman to do put with different parameters and getting the code in python
url1 = "https://sv443.net/jokeapi/v2/submit"

payloadd = "{\r\n    \"formatVersion\": 2,\r\n    \"category\": \"Programming\",\r\n    \"type\": \"twopart\",\r\n    \"flags\": {\r\n        \"nsfw\": true,\r\n        \"religious\": true,\r\n        \"political\": true,\r\n        \"racist\": false,\r\n        \"sexist\": false\r\n    },\r\n    \"setup\": \"That is the way it has always been \",\r\n    \"delivery\": \"It just change\"\r\n}"
headers = {
'Content-Type': 'application/json'
}

response = requests.request("PUT", url1, headers=headers, data = payloadd)

print(response.text.encode('utf8'))
#Same scenario but different parameters
url = "https://sv443.net/jokeapi/v2/submit"

payload = "{\r\n    \"formatVersion\": 2,\r\n    \"category\": \"Miscellaneous\",\r\n    \"type\": \"single\",\r\n    \"flags\": {\r\n        \"nsfw\": true,\r\n        \"religious\": true,\r\n        \"political\": false,\r\n        \"racist\": false,\r\n        \"sexist\": false\r\n    },\r\n    \"joke\": \"1+1 =11\\n2+2 =22\\n3+3 =33\\n...\\n10+10=1010\"\r\n}"
headers = {
'Content-Type': 'application/json'
}

response = requests.put( url, headers=headers, data = payload)

print(response.text.encode('utf8'))
#

url = "https://sv443.net/jokeapi/v2/submit"

payload = "{\r\n    \"formatVersion\": 2,\r\n    \"category\": \"Programming\",\r\n    \"type\": \"twopart\",\r\n    \"flags\": {\r\n        \"nsfw\": true,\r\n        \"religious\": true,\r\n        \"political\": true,\r\n        \"racist\": false,\r\n        \"sexist\": false\r\n    },\r\n    \"setup\": \"That is the way it has always been \",\r\n    \"delivery\": \"It just change\"\r\n}"
headers = {
'Content-Type': 'application/json'
}

response = requests.request("PUT", url, headers=headers, data = payload)

print(response.text.encode('utf8'))


# In[1]:


import requests

url = "https://sv443.net/jokeapi/v2/joke/Programming?blacklistFlags=nsfw,religious,political&type=twopart&Category=Programming&idRange=13-16'"

payload = "{\r\n    \"category\": \"Programming\",\r\n    \"type\": \"twopart\",\r\n    \"setup\": \"Why did the programmer quit his job?\",\r\n    \"delivery\": \"Because He didn't get arrays.\",\r\n    \"flags\": {\r\n        \"nsfw\": false,\r\n        \"religious\": false,\r\n        \"political\": false,\r\n        \"racist\": false,\r\n        \"sexist\": false\r\n    },\r\n    \"id\": 16,\r\n    \"error\": false\r\n}"
headers = {
		'Content-Type': 'application/json'
}

response = requests.request("GET", url, headers=headers, data = payload)

print(response.text.encode('utf8'))


# In[ ]:
url = "https://httpbin.org/post"

payload = {}
headers= {}

response = requests.request("POST", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
#view the server’s response headers using a Python dictionary
response.headers



