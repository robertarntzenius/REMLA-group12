import requests
import bs4

URL = 'https://stackoverflow.com/questions'
tab = '?tab=Votes&pagesize=50'

# If you want to go to the next page, append &page=2 to the end of the URL

# Follow stackoverflow guidelines (https://stackoverflow.blog/2014/01/23/stack-exchange-cc-data-now-hosted-by-the-internet-archive/)

page = requests.get(URL + tab)
soup = bs4.BeautifulSoup(page.content, 'html.parser')

# Get the div with the id of 'questions'
questions = soup.find(id='questions')
# Get the div with the class with prefix 'question-summary'
question_summary = questions.find_all('div', id=lambda x: x and x.startswith('question-summary'))

titles = []
tags_list = []

# For each question_summary
for question in question_summary:
    # Find h3 tag
    h3 = question.find('h3')
    # Find a tag
    a = h3.find('a')
    # Add a text to title list
    titles.append(a.text)

    # Find div with name s-post-summary--meta
    meta = question.find('div', class_='s-post-summary--meta-tags')
    # Find a tags
    a_tags = meta.find_all('a')
    tags = []
    # Add a text to tags list
    for a in a_tags:
        tags.append(a.text)
    tags_list.append(tags)

print(titles)
print(tags_list)

