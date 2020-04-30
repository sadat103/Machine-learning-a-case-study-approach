#!/usr/bin/env python
# coding: utf-8

# In[1]:


import graphlab


# In[2]:


#read some product review data


# In[3]:


products = graphlab.SFrame('m_bfaa91c17752f745.frame_idx')


# In[4]:


#Lets explore this data together


# In[5]:


products.head()


# In[6]:


len(products)


# In[7]:


#build the word count  vector for each review


# In[9]:


products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[11]:


products.head()


# In[12]:


graphlab.canvas.set_target('ipynb')


# In[13]:


products['name'].show()


# In[14]:


#explore vullie shophie


# In[15]:


giraffe_reviews = products[products['name']== 'Vulli Sophie the Giraffe Teether']


# In[16]:


len(giraffe_reviews)


# In[19]:


giraffe_reviews['rating'].show(view='Categorical')


# In[20]:


#build a sentiment classifier


# In[22]:


products['rating'].show(view='Categorical')


# In[23]:


#DEfine whats postiitve and negative sentiment
#ignore all 3* reviews 


# In[24]:


products = products[products['rating']!=3]


# In[25]:


#postive sentiment == 4* or 5* reviews
products['sentiment']=products['rating']>=4


# In[26]:


products.head()


# In[ ]:




#Lets train the sentiment classifier
# In[28]:


train_data,test_data = products.random_split(0.8,seed=0)


# In[29]:


sentiment_model = graphlab.logistic_classifier.create(train_data, target = 'sentiment',features=['word_count'],validation_set=test_data)


# In[30]:


sentiment_model.evaluate(test_data , metric = 'roc_curve')


# In[31]:


sentiment_model.show(view='Evaluation')


# In[32]:


#apply the learned model to understand sentiment  giraffe


# In[33]:


giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews,output_type ='probability')


# In[34]:


giraffe_reviews.head()


# In[35]:


##sort revw based on predicted sentiment and explore


# In[37]:


giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[38]:


giraffe_reviews.head()


# In[39]:


giraffe_reviews[0]['review']


# In[40]:


giraffe_reviews[1]['review']


# In[41]:


#show most nnegative reviews


# In[42]:


giraffe_reviews[-1]['review']


# In[43]:


selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']


# In[44]:


def awesome_count(word_count):
    if 'awesome' in word_count:
        return word_count['awesome']
    return 0

products['awesome'] = products['word_count'].apply(awesome_count)


def great_count(word_count):
    if 'great' in word_count:
        return word_count['great']
    return 0
products['great'] = products['word_count'].apply(great_count)

def fantastic_count(word_count):
    if 'fantastic' in word_count:
        return word_count['fantastic']
    return 0
products['fantastic'] = products['word_count'].apply(fantastic_count)

def amazing_count(word_count):
    if 'amazing' in word_count:
        return word_count['amazing']
    return 0

products['amazing'] = products['word_count'].apply(amazing_count)

def love_count(word_count):
    if 'love' in word_count:
        return word_count['love']
    return 0

products['love'] = products['word_count'].apply(love_count)

def horrible_count(word_count):
    if 'horrible' in word_count:
        return word_count['horrible']
    return 0

products['horrible'] = products['word_count'].apply(horrible_count)

def bad_count(word_count):
    if 'bad' in word_count:
        return word_count['bad']
    return 0

products['bad'] = products['word_count'].apply(bad_count)

def terrible_count(word_count):
    if 'terrible' in word_count:
        return word_count['terrible']
    return 0

products['terrible'] = products['word_count'].apply(terrible_count)

def awful_count(word_count):
    if 'awful' in word_count:
        return word_count['awful']
    return 0

products['awful'] = products['word_count'].apply(awful_count)

def wow_count(word_count):
    if 'wow' in word_count:
        return word_count['wow']
    return 0

products['wow'] = products['word_count'].apply(wow_count)

def hate_count(word_count):
    if 'hate' in word_count:
        return word_count['hate']
    return 0

products['hate'] = products['word_count'].apply(hate_count)


# In[45]:


products.head()


# In[46]:


for word in selected_words:
    print('{0} : {1}'.format(word, products[word].sum()))


# In[47]:


train_data,test_data = products.random_split(.8, seed=0)


# In[51]:


selected_words_model = graphlab.logistic_classifier.create(train_data, target = 'sentiment',features=selected_words,validation_set=test_data)


# In[52]:


coef = selected_words_model['coefficients']


# In[53]:


coef.sort('value', ascending=False)


# In[55]:


coef.sort('value', ascending=True)


# In[58]:


selected_words_model.evaluate(test_data)


# In[57]:


sentiment_model.evaluate(test_data)



# In[59]:


diaper_champ_reviews = products[products['name']== 'Baby Trend Diaper Champ']


diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews,output_type ='probability')


# In[77]:


diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment2', ascending=True)


# In[78]:


diaper_champ_reviews.head()


# In[65]:


diaper_champ_reviews['predicted_sentiment2']=selected_words_model.predict(diaper_champ_reviews, output_type='probability')


# In[66]:


diaper_champ_reviews.head()


# In[75]:


diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment2', ascending=True)


# In[76]:


diaper_champ_reviews.head()


# In[69]:


diaper_champ_reviews[0]['review']


# In[72]:


diaper_champ_reviews[0]['awesome']


# In[1]:


diaper_champ_reviews['predicted_sentiment2'].mean()


# In[ ]:




