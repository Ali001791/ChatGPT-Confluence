import os
import nltk
import openai
import numpy as np
import pandas as pd
from atlassian import Confluence
from bs4 import BeautifulSoup
from transformers import GPT2TokenizerFast

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

CONFLUENCE_URL = 'https://zerowing.atlassian.net/'
CONFLUENCE_SPACE = 'Recipes'
CONFLUENCE_USER = "zerowing@gmail.com"
CONFLUENCE_PASSWORD = 'API_Key_For_Confluence'
OPENAI_API_KEY =  'OPENAI_API_KEY'
EMBEDDING_MODEL = 'text-search-ada-doc-001'
COMPLETIONS_MODEL = "gpt-3.5-turbo" 

openai.api_key = OPENAI_API_KEY

def connect_to_Confluence():
    url = CONFLUENCE_URL
    username = CONFLUENCE_USER
    password  = CONFLUENCE_PASSWORD
    confluence = Confluence(
        url=url,
        username=username,
        password=password,
        cloud=True)

    return confluence

def get_all_pages(confluence, space=CONFLUENCE_SPACE):
    keep_going = True
    start = 0
    limit = 100
    pages = []
    while keep_going:
        results = confluence.get_all_pages_from_space(space, start=start, limit=100, status=None, expand='body.storage', content_type='page')
        pages.extend(results)
        if len(results) < limit:
            keep_going = False
        else:
            start = start + limit
    return pages

def get_embeddings(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]

def get_max_num_tokens():
    return 2046

def collect_title_body_embeddings(pages, save_csv=True):
    collect = []
    for page in pages:
        title = page['title']
        link = CONFLUENCE_URL + '/wiki/spaces/'+CONFLUENCE_SPACE+'/pages/' + page['id']
        htmlbody = page['body']['storage']['value']
        htmlParse = BeautifulSoup(htmlbody, 'html.parser')
        body = []
        for para in htmlParse.find_all("p"):
            sentence = para.get_text()
            tokens = nltk.tokenize.word_tokenize(sentence)
            token_tags = nltk.pos_tag(tokens)
            tags = [x[1] for x in token_tags]
            if any([x[:2] == 'VB' for x in tags]):
                if any([x[:2] == 'NN' for x in tags]):
                    body.append(sentence)
        body = '. '.join(body)
        tokens = tokenizer.encode(body)
        collect += [(title, link, body, len(tokens))]
    DOC_title_content_embeddings = pd.DataFrame(collect, columns=['title', 'link', 'body', 'num_tokens'])
    DOC_title_content_embeddings = DOC_title_content_embeddings[DOC_title_content_embeddings.num_tokens<=get_max_num_tokens()]
    doc_model = EMBEDDING_MODEL
    DOC_title_content_embeddings['embeddings'] = DOC_title_content_embeddings.body.apply(lambda x: get_embeddings(x, doc_model))

    if save_csv:
        DOC_title_content_embeddings.to_csv('DOC_title_content_embeddings.csv', index=False)

    return DOC_title_content_embeddings

def update_internal_doc_embeddings():
    confluence = connect_to_Confluence()
    pages = get_all_pages(confluence, space=CONFLUENCE_SPACE)
    DOC_title_content_embeddings= collect_title_body_embeddings(pages, save_csv=True)
    return DOC_title_content_embeddings

def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, doc_embeddings: pd.DataFrame):
    query_model = EMBEDDING_MODEL
    query_embedding = get_embeddings(query, model=query_model)
    doc_embeddings['similarity'] = doc_embeddings['embeddings'].apply(lambda x: vector_similarity(x, query_embedding))
    doc_embeddings.sort_values(by='similarity', inplace=True, ascending=False)
    doc_embeddings.reset_index(drop=True, inplace=True)

    return doc_embeddings

def construct_prompt(query, doc_embeddings):
    MAX_SECTION_LEN = get_max_num_tokens()
    SEPARATOR = "\n* "
    separator_len = len(tokenizer.tokenize(SEPARATOR))

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_links = []

    for section_index in range(len(doc_embeddings)):
        document_section = doc_embeddings.loc[section_index]

        chosen_sections_len += document_section.num_tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.body.replace("\n", " "))
        chosen_sections_links.append(document_section.link)

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    prompt = header + "".join(chosen_sections) + "\n\n Q: " + query + "\n A:"

    return (prompt,  chosen_sections_links)

def internal_doc_chatbot_answer(query, DOC_title_content_embeddings):
    DOC_title_content_embeddings = order_document_sections_by_query_similarity(query, DOC_title_content_embeddings)
    prompt, links = construct_prompt(query, DOC_title_content_embeddings)

    messages = [
        {"role": "system", "content": "You answer questions about the Recipes space."},
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
        model=COMPLETIONS_MODEL,
        messages=messages,
        temperature=0
    )

    output = response["choices"][0]["message"]["content"].strip(" \n")

    return output, links
