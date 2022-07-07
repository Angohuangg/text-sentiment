#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, render_template


# In[2]:


from transformers import pipeline


# In[3]:


classifier = pipeline('sentiment-analysis', "mrm8488/bert-small-finetuned-squadv2")


# In[4]:


app = Flask(__name__)

@app.route("/",methods = ["GET","POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        print(text)
        R = classifier(text)
        return(render_template("index.html",result = R))
    else:
        return(render_template("index.html",result = "2"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




