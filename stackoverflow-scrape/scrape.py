import argparse
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
    :param tab: 'Votes', 'Active', 'Newest', 'Bountied', 'Unanswered', 'Frequent'
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
        q_df = get_test_data(soup)
        # Concatenate dataframe
        df = pd.concat([df, q_df], ignore_index=True)
        # Print progress
        print('Page ' + str(i + 1) + ' of ' + str(pages))
    # Return dataframe
    return df


def store_as_tsv(df, filename):
    """
    Store dataframe as tsv according to guidelines
    (https://stackoverflow.blog/2014/01/23/stack-exchange-cc-data-now-hosted-by-the-internet-archive/)
    :param df: dataframe
    :param filename: filename
    :return: None
    """
    # Store as tsv in root directory
    df.to_csv(filename, sep='\t', index=False)


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
    q_df = get_questions(question_summary)
    # Return the questions dataframe
    return q_df


def get_questions(questions):
    """
    Get questions dataframe from questions html object
    :param questions: html object
    :return: questions dataframe
    """
    titles = get_titles(questions)
    tags_list = get_tags(questions)
    hrefs = get_hrefs(questions)
    authors = get_authors(questions)
    author_urls = get_author_urls(questions)
    # Create dataframe
    q_df = pd.DataFrame()
    # Add lists to dataframe
    q_df['title'] = titles
    q_df['tags'] = tags_list
    q_df['href'] = hrefs
    q_df['author'] = authors
    q_df['author_url'] = author_urls
    # Return dataframe
    return q_df


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
            text = a_tags.text
            # Remove trailing whitespace
            text = text.strip()
            # Check if text starts with a number
            if text[0].isdigit():
                authors.append("Anonymous")
            else:
                authors.append(text)
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


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--amount', type=int, default=100, help='Amount of questions to scrape (multiple of 50)')
    parser.add_argument('-t', '--tab', type=str, default='Votes', help='Stackoverflow tab to scrape, choice of: '
                                                                       'Votes, Active, '
                                                                       'Newest, Bountied, Unanswered, Frequent')
    parser.add_argument('-f', '--filename', type=str, default='questions.tsv', help='Filename to store questions')
    args = parser.parse_args()
    # Get questions
    questions_df = run_parse(args.amount, args.tab)
    # Store as tsv
    store_as_tsv(questions_df, args.filename)
    # Print message
    print('Questions scraped and stored as ' + args.filename)
