import os
from glob import glob
import xml.etree.ElementTree as ET
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import lda

data_dirs = ['/home/wcroughan/glasses_data/ami_corpus']
data_dir = None
for dd in data_dirs:
    if os.path.exists(dd):
        data_dir = dd
        break
if data_dir is None:
    raise Exception("Couldn't find any of the listed data directories")
output_dir = os.path.join(os.getcwd(), "output")


def importWords(meeting_name):
    wordFiles = glob(os.path.join(data_dir, 'words', meeting_name + ".*.words.xml"))
    names = []
    trees = []
    for wf in wordFiles:
        trees.append(ET.parse(wf))
        sufidx = len(wf) - len(".words.xml")
        namestart = wf.rfind(".", 0, sufidx)
        names.append(wf[namestart+1:sufidx])

    return trees, names


def makeTranscript(roots, names, speakerSwitchDelay=2, skipTruncatedWords=True):
    numSpeakers = len(roots)
    words = [""] * numSpeakers
    times = np.zeros((numSpeakers, 1))
    numDone = 0

    iters = [r.iter() for r in roots]
    for i, it in enumerate(iters):
        n = next(it)
        n = next(it)
        while n.tag != "w":
            n = next(it)
        words[i] = n.text
        times[i] = n.attrib['starttime']

    lastSpeaker = -1
    transcript = []
    i = 0
    while numDone < len(roots):
        # if i % 10000 == 0:
        # print(i, lastSpeaker)
        i += 1

        nextSpeaker = np.argwhere(times == np.min(times))[0][0]
        if lastSpeaker != nextSpeaker and times[lastSpeaker] - times[nextSpeaker] < speakerSwitchDelay:
            # if lastspeaker still talking, don't switch yet
            nextSpeaker = lastSpeaker

        if lastSpeaker == nextSpeaker:
            transcript[-1][1].append(words[nextSpeaker])
        else:
            transcript.append([names[nextSpeaker], [words[nextSpeaker]]])

        if transcript[-1][1][-1] is None:
            raise Exception("found weird data at iter {}".format(i))

        try:
            n = next(iters[nextSpeaker])
            while n.tag != "w" or ('trunc' in n.attrib.keys() and n.attrib['trunc'] == "true"):
                n = next(iters[nextSpeaker])
            words[nextSpeaker] = n.text
            times[nextSpeaker] = n.attrib['starttime']
        except StopIteration:
            words[nextSpeaker] = ""
            times[nextSpeaker] = np.inf
            numDone += 1

        lastSpeaker = nextSpeaker

    sentences = [l[0] + ": " + ' '.join(l[1]) for l in transcript]
    all_concat = '\n'.join(sentences)
    return all_concat


def getRawSpeakerTranscript(speakerRoot, start=0, stop=np.inf, skipTruncatedWords=True):
    def filt(w): return ((not skipTruncatedWords) or ('trunc' not in w.attrib.keys() or w.attrib['trunc'] == "false")) and float(
        w.attrib['starttime']) >= start and float(w.attrib['starttime']) <= stop
    return ' '.join([w.text for w in speakerRoot.iter('w') if filt(w)])


def gramsAsList(speakerRoot, ngram=1, start=0, stop=300):
    """
    returns all ngrams from speakerRoot's dialog from time start to stop (in secs)
    """
    res = []
    gram = []
    for w in speakerRoot.iter('w'):
        time = float(w.attrib['starttime'])
        if time < start:
            continue
        if time > stop:
            break

        if 'trunc' in w.attrib.keys() and w.attrib['trunc'] == "true":
            continue

        if 'punc' in w.attrib.keys() and w.attrib['punc'] == "true":
            gram = []
            continue

        word = str(w.text)
        gram.append(word)
        if len(gram) < ngram:
            continue

        if len(gram) > ngram:
            p = gram.pop(0)
            # print(p)
            assert len(gram) == ngram

        res.append(gram.copy())

    return res


if __name__ == "__main__":
    allMeetingFileNames = sorted(list(set(glob(os.path.join(data_dir, 'words', "EN*.words.xml")))))
    allMeetingNames = sorted(
        list(set([s[s.rfind("/")+1:-len(".A.words.xml")] for s in allMeetingFileNames])))
    print(allMeetingNames, "{} meetings".format(len(allMeetingNames)))
    words = []
    # roots = []
    for meeting in allMeetingNames:
        trees, names = importWords(meeting)
        thisroots = [t.getroot() for t in trees]
        # roots = roots + thisroots
        words.append(' ; '.join([getRawSpeakerTranscript(s) for s in thisroots]))

    # meeting_name = 'EN2001a'
    # trees, names = importWords(meeting_name)
    # roots = [t.getroot() for t in trees]
    # print("Making transcript")
    # transcript = makeTranscript(roots, names)
    # output_file = os.path.join(output_dir, "transcript_{}.txt".format(meeting_name))
    # with open(output_file, "w") as f:
    #     f.write(transcript)

    # allScripts = []
    # for i, r in enumerate(roots):
    #     print("Starting cleanup for speaker {}".format(names[i]))
    #     grams1 = gramsAsList(r, ngram=1, start=0, stop=300)
    #     grams2 = gramsAsList(r, ngram=2, start=0, stop=300)
    #     output_file = os.path.join(output_dir, "grams_{}_{}.txt".format(meeting_name, names[i]))
    #     with open(output_file, "w") as f:
    #         f.write('\n'.join([' '.join(g) for g in grams2]))

    # words = [getRawSpeakerTranscript(s, stop=30) for s in roots]
    # print(words)
    # words = [getRawSpeakerTranscript(s) for s in roots]

    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english',
                                 token_pattern=r"(?u)\b\w\w\w\w+\b")

    intervalWordVecs = vectorizer.fit_transform(words)

    # intervalWordVecs = np.zeros((0, len(vectorizer.get_feature_names())), dtype=np.int64)

    # # print(vectorizer.get_feature_names())
    # # interval = 60
    # intervalWordVecs = np.zeros((0, len(vectorizer.get_feature_names())), dtype=np.int64)
    # t1 = 0
    # while True:
    #     # print(i)
    #     t2 = t1 + interval
    #     wordlist = [getRawSpeakerTranscript(s, start=t1, stop=t2) for s in roots]
    #     ws = ' '.join(wordlist)
    #     if len(ws) == len(wordlist) - 1:
    #         break
    #     x = vectorizer.transform([ws])
    #     intervalWordVecs = np.vstack((intervalWordVecs, x.toarray()))
    #     # print(ws)
    #     # print(x.toarray().shape)
    #     # print(np.sum(x.toarray()))

    #     t1 = t2

    # # print(intervalWordVecs.shape)
    # # print(intervalWordVecs.dtype)
    # # print(intervalWordVecs[0:3, 0:3])

    numTopics = 15

    ldaModel = lda.LDA(n_topics=numTopics, n_iter=2000, random_state=1)
    x = ldaModel.fit_transform(intervalWordVecs)

    vocab = np.array(vectorizer.get_feature_names())

    topic_word = ldaModel.topic_word_
    n_top_words = 8
    for i, topic_dist in enumerate(topic_word):
        topic_words = vocab[np.argsort(topic_dist)][: - (n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' ; '.join(topic_words)))

    doc_topic = ldaModel.doc_topic_

    print(doc_topic.shape)
    bintop = np.argmax(doc_topic, axis=1)
    print(bintop.shape)

    h, _ = np.histogram(bintop, bins=numTopics)
    print(h)
