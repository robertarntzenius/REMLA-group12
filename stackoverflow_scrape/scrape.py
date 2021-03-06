"""Module to scrape stackoverflow."""
import argparse
import os
from collections import Counter

import bs4
import pandas as pd
import requests

URL = "https://stackoverflow.com/questions"
tags = ["python", "java", "php"]
tags_count = []


# Follow stackoverflow guidelines
# (https://stackoverflow.blog/2014/01/23/stack-exchange-cc-data-now-hosted-by-the-internet-archive/)


def run_parse(amount, tab="Votes"):
    """
    Run stackoverflow scraper
    :param amount: multiple of 50
    :param tab: 'Votes', 'Active', 'Newest', 'Bountied', 'Unanswered', 'Frequent'
    :return: pandas dataframe with questions
    """
    # Count the number of pages
    pages = amount // 50
    # Create empty dataframe
    dataframe = pd.DataFrame()
    tags_url_section = make_tags_url_section()
    # Loop through pages
    for i in range(pages):
        # Get the page
        page = requests.get(
            URL + tags_url_section + "?tab=" + tab + "&pagesize=50&page=" + str(i + 1)
        )
        # Get the soup
        soup = bs4.BeautifulSoup(page.content, "html.parser")
        # Get the questions dictionary
        q_df = get_test_data(soup)
        # Concatenate dataframe
        dataframe = pd.concat([dataframe, q_df], ignore_index=True)
        # Print progress
        print("Page " + str(i + 1) + " of " + str(pages))
    # Return dataframe
    return dataframe


def store_as_tsv(dataframe, directory):
    """
    Store dataframe as tsv according to guidelines
    (https://stackoverflow.blog/2014/01/23/stack-exchange-cc-
    data-now-hosted-by-the-internet-archive/)
    :param dataframe: dataframe
    :param directory: directory to store tsvs
    :return: None
    """
    # If directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Store as tsv in root directory
    dataframe.to_csv(directory + "/all_data.tsv", sep="\t", index=False)
    # Get title and tags columns
    title_tags = dataframe[["title", "tags"]]
    # Split title_tags into train (0.8*0.8), test (0.2) and validation (0.8*0.2)
    train = title_tags[: int(len(title_tags) * 0.8)]
    test = title_tags[int(len(title_tags) * 0.8) : int(len(title_tags) * 0.9)]
    validation = train[int(len(train) * 0.8) :]
    train = train[: int(len(title_tags) * 0.8)]
    # Store as tsvs
    train.to_csv(directory + "/train.tsv", sep="\t", index=False)
    test[["title"]].to_csv(directory + "/test.tsv", sep="\t", index=False)
    validation.to_csv(directory + "/validation.tsv", sep="\t", index=False)


def get_test_data(soup):
    """
    Get test data from soup page
    :param soup: bs4 soup page
    :return: pandas dataframe with questions
    """
    # Get the div with the id of 'questions'
    questions = soup.find(id="questions")
    # Get the div with the class with prefix 'question-summary'
    question_summary = questions.find_all(
        "div", id=lambda x: x and x.startswith("question-summary")
    )
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
    count_tags(questions)
    # Create dataframe
    q_df = pd.DataFrame()
    # Add lists to dataframe
    q_df["title"] = titles
    q_df["tags"] = tags_list
    q_df["href"] = hrefs
    q_df["author"] = authors
    q_df["author_url"] = author_urls
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
        head_3 = question.find("h3")
        # Find a tag
        a_html = head_3.find("a")
        # Add a text to title list
        titles.append(a_html.text)
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
        head_3 = question.find("h3")
        # Find a tag
        a_html = head_3.find("a")
        # Add a href to href list
        hrefs.append("https://stackoverflow.com" + a_html["href"])
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
        meta = question.find("div", class_="s-post-summary--meta-tags")
        # Find a tags
        a_tags = meta.find_all("a")
        tag = []
        # Add a text to tags list
        for a_html in a_tags:
            # If tag is in tags list, add it
            if a_html.text in tags:
                tag.append(a_html.text)
        tags_list.append(tag)
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
        meta = question.find("div", class_="s-user-card--info")
        # Find a tags
        a_tags = meta.find("a")
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
            authors.append("Anonymous")
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
        meta = question.find("div", class_="s-user-card--link")
        # Find a tags
        a_tags = meta.find("a")
        if a_tags is not None:
            # Add a href to author_urls list
            author_urls.append("https://stackoverflow.com" + a_tags["href"])
        else:
            author_urls.append("Anonymous")
    return author_urls


def count_tags(questions):
    """
    Count tags in questions html object
    :param questions: html object
    :return: list of tags
    """
    # tags_count list with tuples (tag, count)
    for question in questions:
        # Find div with name s-post-summary--meta
        meta = question.find("div", class_="s-post-summary--meta-tags")
        # Find a tags
        a_tags = meta.find_all("a")
        # Loop over a_tags, if tag is not in tags_count tuple, add it
        for a_html in a_tags:
            tags_count.append(a_html.text)
    return tags_count


def get_tags_count():
    """
    Get count of tags.
    :return: count
    """
    count = Counter(tags_count)
    return count


def make_tags_url_section():
    """
    Make tags url section from tags list
    :param t: list of tags
    :return: tags url section
    """
    tags_url_section = "/tagged/"
    # Loop over tags except last one
    for tag in tags[:-1]:
        tags_url_section += tag + " or "
    # Add last tag
    tags_url_section += tags[-1]
    return tags_url_section


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--amount",
        type=int,
        default=100,
        help="Amount of questions to scrape (multiple of 50)",
    )
    parser.add_argument(
        "-t",
        "--tab",
        type=str,
        default="Votes",
        help="Stackoverflow tab to scrape, choice of: "
        "Votes, Active, "
        "Newest, Bountied, Unanswered, Frequent",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="generated",
        help="Directory to store question sets",
    )
    args = parser.parse_args()
    # Get questions
    questions_df = run_parse(args.amount, args.tab)
    # Store as tsv
    store_as_tsv(questions_df, args.directory)
    print(get_tags_count())
    # Print message
    print("Questions scraped and in " + args.directory + " directory.")
