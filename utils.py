import torch
from dictionary import Dictionary
from logging import getLogger
import io
import numpy as np

logger = getLogger()


def select_subset(word_list, max_vocab):
    """
    Select a subset of words to consider, to deal with words having embeddings
    available in different casings. In particular, we select the embeddings of
    the most frequent words, that are usually of better quality.
    """
    word2id = {}
    indexes = []
    for i, word in enumerate(word_list):
        word = word.lower()
        if word not in word2id:
            word2id[word] = len(word2id)
            indexes.append(i)
        if max_vocab > 0 and len(word2id) >= max_vocab:
            break
    assert len(word2id) == len(indexes)
    return word2id, torch.LongTensor(indexes)

def load_embeddings(params, source, full_vocab=False):
    """
    Reload pretrained embeddings.
    - `full_vocab == False` means that we load the `params.max_vocab` most frequent words.
      It is used at the beginning of the experiment.
      In that setting, if two words with a different casing occur, we lowercase both, and
      only consider the most frequent one. For instance, if "London" and "london" are in
      the embeddings file, we only consider the most frequent one, (in that case, probably
      London). This is done to deal with the lowercased dictionaries.
    - `full_vocab == True` means that we load the entire embedding text file,
      before we export the embeddings at the end of the experiment.
    """
    assert type(source) is bool and type(full_vocab) is bool
    emb_path = params.src_emb if source else params.tgt_emb
    if emb_path.endswith('.pth'):
        return load_pth_embeddings(params, source, full_vocab)
    if emb_path.endswith('.bin'):
        return load_bin_embeddings(params, source, full_vocab)
    if params.dataset == 'vecmap':
        return read_txt_embeddings_2(params, source, full_vocab)
    else:
        return read_txt_embeddings(params, source, full_vocab)

def load_pth_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a PyTorch binary file.
    """
    # reload PyTorch binary file
    lang = params.src_lang if source else params.tgt_lang
    data = torch.load((params.src_emb if source else params.tgt_emb), map_location=torch.device('cuda') if params.cuda else torch.device('cpu'))
    max_vocab = params.max_vocab_A if source else params.max_vocab_B
    dico = data['dico']
    embeddings = data['vectors']
    assert dico.lang == lang
    assert embeddings.size() == (len(dico), params.emb_dim)
    logger.info("Loaded %i pre-trained word embeddings." % len(dico))

    # select a subset of word embeddings (to deal with casing)
    if not full_vocab:
        word2id, indexes = select_subset([dico[i] for i in range(len(dico))], max_vocab)
        id2word = {v: k for k, v in word2id.items()}
        dico = Dictionary(id2word, word2id, lang)
        embeddings = embeddings[indexes]

    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings


def load_bin_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a fastText binary file.
    """
    # reload fastText binary file
    lang = params.src_lang if source else params.tgt_lang
    model = load_fasttext_model(params.src_emb if source else params.tgt_emb)
    max_vocab = params.max_vocab_A if source else params.max_vocab_B
    words = model.get_labels()
    assert model.get_dimension() == params.emb_dim
    logger.info("Loaded binary model. Generating embeddings ...")
    embeddings = torch.from_numpy(np.concatenate([model.get_word_vector(w)[None] for w in words], 0))
    logger.info("Generated embeddings for %i words." % len(words))
    assert embeddings.size() == (len(words), params.emb_dim)

    # select a subset of word embeddings (to deal with casing)
    if not full_vocab:
        word2id, indexes = select_subset(words, max_vocab)
        embeddings = embeddings[indexes]
    else:
        word2id = {w: i for i, w in enumerate(words)}
    id2word = {i: w for w, i in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)

    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings

def read_txt_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    lang = params.src_lang if source else params.tgt_lang
    emb_path = params.src_emb if source else params.tgt_emb
    max_vocab = params.max_vocab_A if source else params.max_vocab_B
    _emb_dim_file = params.emb_dim
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                #assert _emb_dim_file == int(split[1])
            else:
                word, vect = line.rstrip().split(' ', 1)
                if not full_vocab:
                    word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:
                        logger.warning("Word '%s' found twice in %s embedding file"
                                       % (word, 'source' if source else 'target'))
                else:
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
            if max_vocab > 0 and len(word2id) >= max_vocab and not full_vocab:
                break

    assert len(word2id) == len(vectors)
    logger.info("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)
    embeddings = np.concatenate(vectors, 0)
    
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if (params.cuda and not full_vocab) else embeddings
    print(embeddings.size())
 
    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings


def read_txt_embeddings_2(params, source, full_vocab):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    lang = params.src_lang if source else params.tgt_lang
    emb_path = params.src_emb if source else params.tgt_emb
    max_vocab = params.max_vocab_A if source else params.max_vocab_B
    _emb_dim_file = params.emb_dim
    file = open(emb_path, encoding='utf-8', errors='surrogateescape')
    header = file.readline().split(' ')
    count = int(header[0]) if max_vocab <= 0 else min(max_vocab, int(header[0]))
    dim = int(header[1])
    #words = []
    #word2id = {}
    #matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        word, vect = file.readline().split(' ', 1)
        if word in word2id:
            print(word, " is duplicate")
        else:
            word2id[word] = len(word2id)
            vect = np.fromstring(vect, sep=' ')
            if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
            vectors.append(vect[None])
            

    assert len(word2id) == len(vectors)
    logger.info("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)
    embeddings = np.concatenate(vectors, 0)
    
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if (params.cuda and not full_vocab) else embeddings
     
    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings