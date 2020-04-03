#python2
# enter the code all-together
# https://stackoverflow.com/questions/47559098/is-there-any-way-to-get-abstracts-for-a-given-list-of-pubmed-ids

from Bio import Entrez

Entrez.email = 'taojincs@gmail.com'

# pmids = [17284678,9997] # read from a file
with open('303-train.txt', 'r') as f:
    pmids = [line.strip() for line in f]

handle = Entrez.efetch(db="pubmed", id=','.join(map(str, pmids)),
                       rettype="xml", retmode="text")
records = Entrez.read(handle)
#abstracts = [pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
#             for pubmed_article in records['PubmedArticle']]
abstracts = [pubmed_article['MedlineCitation']['Article']['Abstract'] ['AbstractText'][0]
             for pubmed_article in records['PubmedArticle'] if 'Abstract' in
             pubmed_article['MedlineCitation']['Article'].keys()]
abstract_dict = {}
without_abstract = []

for pubmed_article in records['PubmedArticle']:
    pmid = int(str(pubmed_article['MedlineCitation']['PMID']))
    article = pubmed_article['MedlineCitation']['Article']
    if 'Abstract' in article:
        abstract = article['Abstract']['AbstractText'][0]
        abstract_dict[pmid] = abstract
    else:
       without_abstract.append(pmid)

#print(abstract_dict)
#print(without_abstract)

with open('303-train-pub.txt','w') as log:
    for value in abstract_dict.values():
        print(value)
        log.write('{}\n'.format(value.encode('utf-8')))  # for special char



##### the update version will work

from Bio import Entrez
import time
Entrez.email = 'taojincs@gmail.com'
with open('269-train.txt', 'r') as f:
    pmids = [line.strip() for line in f]

handle = Entrez.efetch(db="pubmed", id=','.join(map(str, pmids)),
                       rettype="xml", retmode="text")
records = Entrez.read(handle)
abstracts = [pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]  if 'Abstract' in pubmed_article['MedlineCitation']['Article'].keys() else pubmed_article['MedlineCitation']['Article']['ArticleTitle']  for pubmed_article in records['PubmedArticle']]
abstract_dict = dict(zip(pmids, abstracts))
#print abstract_dict
with open('269-train-pub.txt','w') as log:
    for value in abstract_dict.values():
        #print(value)
        log.write('{}\n'.format(value.encode('utf-8')))  # for special char
