'''
We're gonna try to do latent dirichlet allocation with variational Bayes

The topic learning will use the model
alpha -> theta -> z -> w <- beta <- eta

where w is observed, alpha and eta are parameters (input), and theta, z, and beta are learned

Topic prediction will probably use the model
alpha -> theta -> z -> w <- beta

where beta is the learned beta (topic-word distribution)

NOTE: ngtj - a lot of this code is derived from https://github.com/blei-lab/onlineldavb
           - I tried not to copy things directly because I really wanted to understand what was going on though
           - more and more I'm understanding why they choose to use classes instead of just native objects
           - here's the paper that OnlineLDA implements: https://proceedings.neurips.cc/paper_files/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf

+=================================================+
|                    CHANGELOG                    |
+=================================================+
20250602: ngtj - Initial creation
20250603: ngtj - Perplexity
20250607: ngtj - More comments with citations

+=================================================+
|                      USAGE                      |
+=================================================+
STEP 1: Run get_docs() on the bbc news data csv (no inputs needed). 
This will return the vocabulary, word_ids, and word_cts used as inputs for the LDA class methods

STEP 2: Figure out 
- how many topics you want to model (n_topics)
- (optional) the variational hyperparameters alpha, eta
- (optional) step size parameters learning_offset, kappa

STEP 3: (optional) split the document set into train, test

STEP 4: create an instance of the LDA class with your chosen parameters and the train size
        if you chose to separate the data

STEP 5: batch your data and train

I'm not sure how we'll do model validation yet - perplexity...? heuristically lmao looking at the groups
'''
import re
import pandas as pd
import numpy as np
import os
from scipy.special import gammaln, psi
# gammaln = log(abs(gamma(x))) where log is the natural log, abs is absolute value, and gamma is the gamma function
# psi = digamma function

# get the stopwords for parsing later
# if you haven't downloaded the stopwords yet, you need to run the following lines:
    # import nltk
    # nltk.download('stopwords')
# if you have never installed the Natural Language Tool Kit (nltk), you can do
    # pip install nltk
# with pip, or
    # conda install anaconda::nltk
# with conda
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# getting rid of more stopwords because they don't mean anything and kept showing up
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'said', 'would', 
                   'also', 'one', 'could', 'bn', 'bbc', 'like', 'mr', 'mrs',
                   'next', 'says', 'told', 'make', 'way', 'get', 'g',
                   'however'])

def get_docs(file_name:str="bbc-news-data.csv"):
    '''
    read documents and return word ids and counts for each document

    this is basically just the whole "corpus" from onlineldavb

    returns:
    - vocab: a dictionary with word:token pairs
        {minister : 0, 
        sport : 1, 
        year : 2,}
    - word_ids: a list of length D, where each entry is a list with unique word tokens from the vocab for that document
        document says something like "minister year sport sport sport" -> word_id[d] = [0, 1, 2]
    - word_cts: a corresponding list of length D, where each entry is how many times that word appeared in the document
        document says something like "minister year sport sport sport" -> word_cts[d] = [1, 1, 3]
    '''
    corpus = pd.read_csv(file_name, sep="\t")
    D = len(corpus)

    word_ids = []
    word_cts = []
    vocab = {}
    
    for d in range(D):
        # read and edit the document
        # note that .content is very specific to bbc-news-data.csv because it's a column title
        doc = corpus.iloc[d].content
        doc = doc.lower()
        doc = re.sub(r'-', ' ', doc) # convert dashes to spaces
        doc = re.sub(r'[^a-z ]', '', doc) # convert non-alphabetical words (e.g. numbers) to nothing (keep spaces)
        doc = re.sub(r' +', ' ', doc) # convert double/triple/quadruple+ spaces to single spaces

        words = doc.split()
        docdict = {}
        for word in words:
            # check if the word is good
            if word in vocab.keys():
                wordtoken = vocab[word]
            elif word not in stop_words:
                vocab[word] = len(vocab)
                wordtoken = vocab[word]
            else:
                continue

            # iterate the single document dictionary
            if wordtoken not in docdict.keys():
                docdict[wordtoken] = 0
            docdict[wordtoken] += 1

        # append document-specific information to the corpus
        word_ids.append(list(docdict.keys()))
        word_cts.append(list(docdict.values()))

    return vocab, word_ids, word_cts

def dirichlet_expectation(alpha):
    '''
    For theta ~ Dir(alpha), compute E[log(theta) | alpha]

    https://www.youtube.com/watch?v=smfWKhDcaoA around 5:00
    '''
    # scipy.special.psi is the digamma function
    if len(alpha.shape)==1:
        return psi(alpha) - psi(np.sum(alpha))
    return psi(alpha) - psi(np.sum(alpha,1))[:, np.newaxis]

def parse_docs(doc_list, vocab):
    '''
    convert list of documents (list of strings) to word_ids, word_cts based on the vocab

    copies a bunch of code from get_docs()
    '''
    if type(doc_list) == str:
        doc_list = [doc_list]
    D = len(doc_list)
    word_ids = []
    word_cts = []

    for d in range(D):
        doc = doc_list[d]
        doc = doc.lower()
        doc = re.sub(r'-', ' ', doc) # convert dashes to spaces
        doc = re.sub(r'[^a-z ]', '', doc) # convert non-alphabetical words (e.g. numbers) to nothing (keep spaces)
        doc = re.sub(r' +', ' ', doc) # convert double/triple/quadruple+ spaces to single spaces

        words = doc.split()
        docdict = {}
        for word in words:
            # check if the word is good
            if word in vocab.keys():
                wordtoken = vocab[word]
            else:
                # we will not be editing the vocabulary
                continue

            # iterate the single document dictionary
            if wordtoken not in docdict.keys():
                docdict[wordtoken] = 0
            docdict[wordtoken] += 1

        # append document-specific information to the corpus
        word_ids.append(list(docdict.keys()))
        word_cts.append(list(docdict.values()))
    
    return word_ids, word_cts

class LDA:
    '''
    Implements Latent Dirichlet Allocation. Based on the OnlineLDA class from https://github.com/blei-lab/onlineldavb/blob/master/onlineldavb.py

    Arguments:
    - vocab: the dictionary of words to recognize; get_docs() will generate this
    - n_classes: the number of topics to find
    - n_docs: the number of documents in the corpus
    - alpha: hyperparameter for prior on weight vectors theta
    - eta: hyperparameter for prior on topics beta
    - learning_offset: positive learning parameter for early iteration downweights
    - kappa: exponential decay learning rate; according to OnlineLDA this should be in (0.5, 1] to guarantee asymptotic convergence
    '''

    def __init__(self, vocab, n_topics, n_docs, alpha=0.1, eta=0.01, learning_offset=1, kappa=0.7, maxiters=100, thres=1e-3):
        '''
        Arguments:
        - vocab: the dictionary of words to recognize; get_docs() will generate this
        - n_classes: the number of topics to find
        - n_docs: the number of documents in the corpus
        - alpha: hyperparameter for prior on weight vectors theta
        - eta: hyperparameter for prior on topics beta
        - learning_offset: positive learning parameter for early iteration downweights
        - kappa: exponential decay learning rate; according to OnlineLDA this should be in (0.5, 1] to guarantee asymptotic convergence
        - maxiters: maximum number of iterations for E step
        - thres: convergence threshold for E step

        Default values for alpha, eta, lr, kappa, maxiters, and thres all come from the OnlineLDA class
        '''

        self._vocab = vocab
        self.n_topics = n_topics
        self.n_docs = n_docs
        self.n_words = len(self._vocab)
        self._alpha = alpha
        self._eta = eta
        self._tao0 = learning_offset
        self._kappa = kappa

        self.maxiters = maxiters
        self.thres = thres
    
        # remember how many times we've updated
        self._updatect = 0

        # initialize variational distribution for topic words randomly q(beta | lambda)
        # in Blei, Ng, and Jordan (2003) they dont mention initialization
        # this initialization comes from onlineldavb
        # I'm not sure the reasoning behind this shape and scale, but with the updates, it shouldn't really matter
        self._lambda = 1 * np.random.gamma(shape=100., scale=1./100., size=(self.n_topics, self.n_words))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)

    def do_e_step(self, word_ids, word_cts):
        '''
        performs the Expectation step in EM for variational Bayesian inferece for LDA

        Arguments:
        - word_ids: list of length [batch size] - contains word ids 
                                                  corresponding to the vocab for each document
        - word_cts: list of length [batch size] - contains the word counts for
                                                  corresponding word ids in each document
        '''

        batch_size = len(word_ids)
        # initialize variational distribution for document topics randomly q(theta | gamma)
        # to be honest, I'm not sure why we're initializing with this gamma distribution
        # maybe it's from the relationship between Dirichlet and gamma?
        # https://en.wikipedia.org/wiki/Dirichlet_distribution#Related_distributions

        # in Blei, Ng, Jordan (2003) section 5.2 (Figure 6) they initialize differently
        # they do gamma = alpha + N/K and phi = 1/K
        # we don't really have N in this case because it was originally the number of words (in the vocabulary)
        # we only have the word counts, and in reality the initialization shouldn't REALLY matter
        # this initialization comes from onlineldavb
        gamma = 1 * np.random.gamma(shape=100., scale=1./100., size=(batch_size, self.n_topics))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        
        # I don't know what this is measuring
        # sufficient statistics for the M step?
        sstats = np.zeros(self._lambda.shape)

        # for each document in the batch, update gamma and phi
        for d in range(batch_size):
            ids = word_ids[d]
            cts = word_cts[d]
            gammad = gamma[d,:]
            Elogthetad = Elogtheta[d,:]
            expElogthetad = expElogtheta[d,:]
            expElogbetad = self._expElogbeta[:, ids]

            # using the rules from Blei, Ng, Jordan (2003) section 5.2 (Figure 6) to update parameters
            '''
            BNJ found that phi_{ni} is proportional to beta_{iw_n} exp(E[log(theta_i) | gamma])
            this implementation uses word ids and counts for each document, so we don't have n or w_n one time per word
            thats alright; we can still use a similar idea

            phi_{ni} is proportional to beta_{in} exp(E[log(theta_i) | gamma])
            exp(E[log (beta_{in}) | lambda]) is VERY similar to beta_{in}; 
            the original derivation is for when you know beta
            we only have our Dirichlet expectation given the variational parameter lambda

            Therefore phi_{ni} is proportional to exp(E[log (beta_{in}) | lambda]) exp(E[log(theta_i) | gamma])

            I've been writing i because BNJ wrote i, but it indexes the topic k

            Since we have exp(E[log(theta_{k}) | gamma]) and exp(E[log(beta_{kn}) | lambda]) for each k, 
            just do a dot product to compute all of the document topic:word probability pairs
            '''
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100 # OnlineLDA added this constant probably for numerical stability
            # no risk of dividing by 0

            # iterate between gamma and phi until convergence
            for it in range(self.maxiters):
                lastgamma = gammad
                # in the paper, gamma's update is written in terms of phi
                # we don't want to compute phi every time, so we'll represent it implicitly
                '''
                ^ those were some words that i wrote because I didn't understand what was going on
                we did compute phi (sort of?)

                in Figure 6 (derived in Appendix A3.1) we see
                gamma_i = alpha_i + sum from n=1 to N of phi_{ni}

                why does expElogthetad * np.dot(cts/phinorm, expElogbetad.T) equal the right thing?
                -> expElogthetad (1 x k) has Dirichlet expectation of theta for this document
                -> cts (n x 1) is the list of word counts for document d
                -> phinorm (1 x n) is the unnormalized phi update
                -> expElogbetad.T (n x k) has Dirichlet expectation of beta for these words

                cts / phinorm (after broadcasting, 1 x n ?) has counts divided by the unnormalized phi
                normally, cts @ expElogbetad.T (1 x k) multiplies the expElogbetad values
                    by how often those words appear in the document. In this case, 
                    we also divide by the phi normalization
                    before multiplying because it would be difficult to separate later
                expElogthetad * (cts / phinorm @ expElogbetad.T) (1 x k) therefore finds the true phi (see the above comments too)
                '''
                gammad = self._alpha + expElogthetad * np.dot(cts/phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                # update phinorm
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

                # convergence check
                if np.mean(abs(gammad-lastgamma)) < self.thres:
                    break

            # contribution of document d to the expected sufficient statistics for the M step
            sstats[:, ids] += np.outer(expElogthetad.T, cts/phinorm)

        # finish the sufficient statistic computation for the M step
        sstats = sstats * self._expElogbeta

        return gamma, sstats
    
    def do_e_step_docs(self, docs):
        '''
        given a batch of documents, estimate parameters 
        gamma controlling the variational distribution over the topic weights
        for each document in the batch

        Arguments:
        - docs: List of D documents. Each must be represented as a string WHATTTTTTTTT THATS INSANEEEEEE

        wait I think this is for predictions so never mind
        
        we never update beta in the E step lol ive calmed down
        '''
        if type(docs)==str:
            docs = [docs]
        
        word_ids, word_cts = parse_docs(docs, self._vocab)

        return self.do_e_step(word_ids, word_cts)
    
    def update_lambda(self, word_ids, word_cts):
        '''
        Performs E step on batch then updates lambda.

        Arguments:
        - word_ids: list of length [batch size] - contains word ids 
                                                  corresponding to the vocab for each document
        - word_cts: list of length [batch size] - contains the word counts for
                                                  corresponding word ids in each document
        '''
        # Blei, Ng, and Jordan (2003) section 5.4
        # how much information do we get from this mini batch?
        # this is a fancy step size
        rhot = (self._tao0 + self._updatect) ** (-self._kappa)

        # perform an E step to update gamma, phi | lambda
        gamma, sstats = self.do_e_step(word_ids, word_cts)

        # estimate held-out likelihood ????? i think now we're referencing a different paper
        bound = self.approx_bound(word_ids, word_cts, gamma)

        # update lambda
        # this looks NOTHING like the algorithm in the blei, ng, jordan paper
        # it looks like what we did in class tho
        self._lambda = self._lambda * (1-rhot) + rhot*(self._eta + self.n_docs * sstats / len(word_ids))

        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = np.exp(self._Elogbeta)
        self._updatect += 1

        return gamma, bound

    def update_lambda_docs(self, docs):
        '''
        performs E step on batch then updates lambda. A wrapper for self.update_lambda()

        Arguments:
        - docs: list of D documents. Each is a string, and any words not in self._vocab will be ignored
        '''
        if type(docs)==str:
            docs = [docs]
        
        word_ids, word_cts = parse_docs(docs, self._vocab)

        return self.update_lambda(word_ids, word_cts)
    
    def approx_bound(self, word_ids, word_cts, gamma, subsampling=True):
        '''
        Estimates variational bound over all documents using only the batch. (ELBO?)
        
        Arguments:
        - word_ids: list of length [batch size] - contains word ids 
                                                  corresponding to the vocab for each document
        - word_cts: list of length [batch size] - contains the word counts for
                                                  corresponding word ids in each document
        - gamma: set of parameters to the variational distribution
                 q(theta) corresponding to the set of documents passed in
        '''
        batch_size = len(word_ids)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)

        # E[log p(docs | theta, beta)]
        for d in range(batch_size):
            ids = word_ids[d]
            cts = np.array(word_cts[d]) # converting to np array for element-wise multiplication later
            phinorm = np.zeros(len(ids))

            for i in range(len(ids)):
                temp = Elogtheta[d,:] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = np.log(sum(np.exp(temp - tmax))) + tmax
            
            score += np.sum(cts * phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += np.sum((self._alpha - gamma) * Elogtheta)
        score += np.sum(gammaln(gamma) - gammaln(self._alpha))
        score += np.sum(gammaln(self._alpha * self.n_topics) - gammaln(np.sum(gamma, 1)))

        # compensate for subsampling (if online)
        if subsampling:
            score = score * self.n_docs / len(word_ids)

        # E[log p(beta | eta) - log q(beta | lambda)]
        score += np.sum((self._eta - self._lambda) * self._Elogbeta)
        score += np.sum(gammaln(self._lambda) - gammaln(self._eta))
        score += np.sum(gammaln(self._eta * self.n_words) - gammaln(np.sum(self._lambda, 1)))

        return score
    
    def perplexity_holdout(self, word_ids, word_cts):
        '''
        Computes the perplexity of a holdout set; used for model validation
        
        Arguments:
        - word_ids: list of length [batch size] - contains word ids 
                                                  corresponding to the vocab for each document
        - word_cts: list of length [batch size] - contains the word counts for
                                                  corresponding word ids in each document
        '''
        gamma, _ = self.do_e_step(word_ids, word_cts)
        bound = self.approx_bound(word_ids, word_cts, gamma, subsampling=False)

        total_words = np.sum([sum(cts) for cts in word_cts])

        return np.exp(-1. * bound / total_words)
    
    def get_topics_for_article(self, article):
        '''
        returns the topic distribution for the input article.

        Arguments:
        - article: a string article or a dictionary with word ids and counts
        '''
        if type(article) == str:
            word_ids, word_cts = parse_docs([article], self._vocab)
        else:
            print(f"Invalid argument for article: Expected type str but got {type(article)} instead.")
            return
        # do E step
        gamma, _ = self.do_e_step(word_ids, word_cts)
        proportions = np.exp(dirichlet_expectation(gamma))
        return proportions / proportions.sum(axis=1, keepdims=True)
        

if __name__=="__main__":
    # STEP 1: run get_docs()
    vocab, word_ids, word_cts = get_docs(file_name="bbc-news-data.csv")

    # STEP 2: define parameters
    n_topics = 5
    alpha = 0.1
    eta = 0.01
    learning_offset = 10
    kappa = 0.75
    maxiters = 100
    thres = 0.001

    # STEP 3: train, test
    # I think this seed is funny lol
    np.random.seed(np.sum([ord(c) for c in "UCI CS 275P LDA"]))
    split = 0.8
    train_idx = np.random.permutation(np.arange(len(word_ids)))[:int(split*len(word_ids))]

    train_word_ids = []
    train_word_cts = []
    test_word_ids = []
    test_word_cts = []
    for idx in range(len(word_ids)):
        if idx in train_idx:
            train_word_ids.append(word_ids[idx])
            train_word_cts.append(word_cts[idx])
        else:
            test_word_ids.append(word_ids[idx])
            test_word_cts.append(word_cts[idx])

    # STEP 4: LDA class instance
    model = LDA(vocab, n_topics, len(train_word_ids), alpha, eta, learning_offset, kappa, maxiters, thres)

    # STEP 5: batch data and train
    # for bbc news data, there are 2225 articles
    # with the seed defined above and an 80% split, there are 1780 articles in train
    # we can try 178 batches of size 10?
    n_batches = 178
    batch_size = 10

    # making a new folder
    cwd = os.getcwd()
    lambdas_path = f'{cwd}/lambdas/'
    if os.path.exists(lambdas_path):
        for root, _, files in os.walk(lambdas_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            os.rmdir(root)

    os.mkdir(lambdas_path)


    for i in range(n_batches):
        print(f"batch [{i+1:3}/{n_batches}]")
        batch_ids = train_word_ids[i*batch_size:(i+1)*batch_size]
        batch_cts = train_word_cts[i*batch_size:(i+1)*batch_size]

        model.update_lambda(batch_ids, batch_cts)
        np.savetxt(f'{lambdas_path}lambda{i:03}.txt', model._lambda.T)

