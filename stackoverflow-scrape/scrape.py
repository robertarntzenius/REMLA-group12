import requests
import bs4
import pandas as pd

URL = 'https://stackoverflow.com/questions'

# Follow stackoverflow guidelines
# (https://stackoverflow.blog/2014/01/23/stack-exchange-cc-data-now-hosted-by-the-internet-archive/)


def run_parse(amount, tab='Votes'):
    """
    Run stackoverflow scraper
    :param amount: multiple of 50
    :param tab: 'Votes', 'Active', 'Newest', 'Bountied', 'Unanswered', 'Frequent', 'Score'
    :return: pandas dataframe with questions
    """
    # Count the number of pages
    pages = amount // 50
    # Create empty dataframe
    df = pd.DataFrame()
    # Loop through pages
    for i in range(pages):
        # Get the page
        page = requests.get(URL + '?tab=' + tab + '&pagesize=50&page=' + str(i + 1))
        # Get the soup
        soup = bs4.BeautifulSoup(page.content, 'html.parser')
        # Get the questions dictionary
        questions_df = get_test_data(soup)
        # Concatenate dataframe
        df = pd.concat([df, questions_df], ignore_index=True)
    # Return dataframe
    return df


def get_test_data(soup):
    """
    Get test data from soup page
    :param soup: bs4 soup page
    :return: pandas dataframe with questions
    """
    # Get the div with the id of 'questions'
    questions = soup.find(id='questions')
    # Get the div with the class with prefix 'question-summary'
    question_summary = questions.find_all('div', id=lambda x: x and x.startswith('question-summary'))
    # Get the questions dataframe
    questions_df = get_questions(question_summary)
    # Return the questions dataframe
    return questions_df


def get_questions(questions):
    """
    Get questions dataframe from questions html object
    :param questions: html object
    :return: questions dataframe
    """
    titles = get_titles(questions)
    hrefs = get_hrefs(questions)
    tags_list = get_tags(questions)
    authors = get_authors(questions)
    author_urls = get_author_urls(questions)
    # Create dataframe
    questions_df = pd.DataFrame()
    # Add lists to dataframe
    questions_df['title'] = titles
    questions_df['href'] = hrefs
    questions_df['tags'] = tags_list
    questions_df['author'] = authors
    questions_df['author_url'] = author_urls
    # Return dataframe
    return questions_df


def get_titles(questions):
    """
    Get titles from questions html object
    :param questions: html object
    :return: list of titles
    """
    titles = []
    for question in questions:
        # Find h3 tag
        h3 = question.find('h3')
        # Find a tag
        a = h3.find('a')
        # Add a text to title list
        titles.append(a.text)
    return titles


def get_hrefs(questions):
    """
    Get hrefs from questions html object
    :param questions: html object
    :return: list of hrefs
    """
    hrefs = []
    for question in questions:
        # Find h3 tag
        h3 = question.find('h3')
        # Find a tag
        a = h3.find('a')
        # Add a href to href list
        hrefs.append('https://stackoverflow.com' + a['href'])
    return hrefs


def get_tags(questions):
    """
    Get tags from questions html object
    :param questions: html object
    :return: questions dataframe
    """
    tags_list = []
    for question in questions:
        # Find div with name s-post-summary--meta
        meta = question.find('div', class_='s-post-summary--meta-tags')
        # Find a tags
        a_tags = meta.find_all('a')
        tags = []
        # Add a text to tags list
        for a in a_tags:
            tags.append(a.text)
        tags_list.append(tags)
    return tags_list


def get_authors(questions):
    """
    Get authors from questions html object
    :param questions: html object
    :return: list of authors
    """
    authors = []
    for question in questions:
        # Find div with name s-user-card--link
        meta = question.find('div', class_='s-user-card--info')
        # Find a tags
        a_tags = meta.find('a')
        if a_tags is not None:
            # Add a text to authors list
            authors.append(a_tags.text)
        else:
            authors.append('Anonymous')
    return authors


def get_author_urls(questions):
    """
    Get author urls from questions html object
    :param questions: html object
    :return: list of author urls
    """
    author_urls = []
    for question in questions:
        # Find div with name s-user-card--link
        meta = question.find('div', class_='s-user-card--link')
        # Find a tags
        a_tags = meta.find('a')
        if a_tags is not None:
            # Add a href to author_urls list
            author_urls.append('https://stackoverflow.com' + a_tags['href'])
        else:
            author_urls.append('Anonymous')
    return author_urls
