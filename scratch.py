import os
from glob import glob
import xml.etree.ElementTree as ET
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import lda
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

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


def makeTranscript(roots, names, speakerSwitchDelay=2, skipTruncatedWords=True, start=0, stop=np.inf):
    numSpeakers = len(roots)
    words = [""] * numSpeakers
    times = np.zeros((numSpeakers, 1))
    numDone = 0

    iters = [r.iter('w') for r in roots]
    for i, it in enumerate(iters):
        try:
            n = next(it)
        except StopIteration:
            print("No words at all for speaker {}".format(names[i]))
            words[i] = ""
            times[i] = np.inf
            numDone += 1
            continue

        times[i] = start-1
        while times[i] < start:
            try:
                n = next(it)
                words[i] = n.text
                times[i] = n.attrib['starttime']
            except StopIteration:
                words[i] = ""
                times[i] = np.inf
                numDone += 1

    lastSpeaker = -1
    transcript = []
    i = 0
    while numDone < len(roots):
        # if i % 10000 == 0:
        # print(i, lastSpeaker)
        i += 1

        nextSpeaker = np.argwhere(times == np.min(times))[0][0]
        if lastSpeaker != -1 and lastSpeaker != nextSpeaker and times[lastSpeaker] - times[nextSpeaker] < speakerSwitchDelay:
            # if lastspeaker still talking, don't switch yet
            nextSpeaker = lastSpeaker

        if lastSpeaker == nextSpeaker:
            try:
                transcript[-1][1].append(words[nextSpeaker])
            except IndexError as ie:
                print(i, transcript)
                raise ie

        else:
            transcript.append([names[nextSpeaker], [words[nextSpeaker]]])

        if transcript[-1][1][-1] is None:
            raise Exception("found weird data at iter {}".format(i))

        try:
            n = next(iters[nextSpeaker])
            while 'trunc' in n.attrib.keys() and n.attrib['trunc'] == "true":
                n = next(iters[nextSpeaker])
            words[nextSpeaker] = n.text
            times[nextSpeaker] = n.attrib['starttime']
            if times[nextSpeaker] > stop:
                raise StopIteration
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
            gram.pop(0)
            # p = gram.pop(0)
            # print(p)
            assert len(gram) == ngram

        res.append(gram.copy())

    return res


def runLDA(wordVecs, numTopics, numIters, plotLikelihood=False):
    model = lda.LDA(n_topics=numTopics, n_iter=numIters, random_state=1)
    model.fit(wordVecs)

    if plotLikelihood:
        plt.plot(model.loglikelihoods_[5:])
        plt.title("model log likelihoods")
        plt.show()

    vocab = np.array(mvecr.get_feature_names())

    topic_word = model.topic_word_
    topic_word_norm = topic_word / np.sum(topic_word, axis=0)
    n_top_words = 8

    doc_topic = model.doc_topic_
    bintop = np.argmax(doc_topic, axis=1)
    h, _ = np.histogram(bintop, bins=numTopics)

    for i, topic_dist in enumerate(topic_word_norm):
        topic_words = vocab[np.argsort(topic_dist)][: - (n_top_words + 1):-1]
        print('Topic {} ({} elements): {}'.format(i, h[i], ' ; '.join(topic_words)))

    return model


if __name__ == "__main__":
    allMeetingFileNames = sorted(list(set(glob(os.path.join(data_dir, 'words', "EN*.words.xml")))))
    allMeetingNames = sorted(
        list(set([s[s.rfind("/")+1:-len(".A.words.xml")] for s in allMeetingFileNames])))
    allWords = []
    roots = []
    names = []
    interval = 120
    for meeting in allMeetingNames:
        print("Getting all words for meeting {}".format(meeting))
        trees, thisnames = importWords(meeting)
        thisroots = [t.getroot() for t in trees]
        names.append(thisnames)
        roots.append(thisroots)
        allWords.append(' ; '.join([getRawSpeakerTranscript(s) for s in thisroots]))

    # mvecr = CountVectorizer(ngram_range=(1, 2), stop_words='english',
        # token_pattern=r"(?u)\b\w\w\w\w+\b")
    mvecr = CountVectorizer(stop_words='english',
                            token_pattern=r"(?u)\b\w\w\w\w+\b")

    meetingWordVecs = mvecr.fit_transform(allWords)
    print("Loaded {} meetings. Meeting-wordvec mtx is size {}".format(len(allMeetingNames), meetingWordVecs.shape))
    print("Running LDA meeting by meeting")
    numTopics = 4
    numIters = 750
    # meetingsModel = runLDA(meetingWordVecs, numTopics, numIters)

    intervalWords = []
    intervalMeetingIdx = []
    intervalMeetingName = []
    intervalWithinMeetingIdx = []
    print("Meeting names: {}".format(allMeetingNames))
    for mi, (meeting, thisroots) in enumerate(zip(allMeetingNames, roots)):
        print("Getting words by interval {} for meeting {}".format(interval, meeting))
        t1 = 0
        wmi = 0
        while True:
            t2 = t1 + interval
            iw = [getRawSpeakerTranscript(s, start=t1, stop=t2) for s in thisroots]
            ws = ' ; '.join(iw)
            if len(ws) == 3*(len(iw) - 1):
                break

            intervalWords.append(ws)
            intervalMeetingIdx.append(mi)
            intervalMeetingName.append(meeting)
            intervalWithinMeetingIdx.append(wmi)
            wmi += 1
            t1 = t2

    # ivecr = CountVectorizer(ngram_range=(1, 2), stop_words='english',
            # token_pattern=r"(?u)\b\w\w\w\w+\b")
    ivecr = CountVectorizer(stop_words='english',
                            token_pattern=r"(?u)\b\w\w\w\w+\b")
    intervalWordVecs = ivecr.fit_transform(intervalWords)

    print("Analyzed {} sec intervals. Interval-wordvec mtx is size {}".format(interval, intervalWordVecs.shape))
    print("Running LDA interval by interval")
    numTopics = 25
    numIters = 3000
    intervalsModel = runLDA(intervalWordVecs, numTopics, numIters, plotLikelihood=True)

    vocab = np.array(mvecr.get_feature_names())

    # doc_topic[doc, topic]
    doc_topic = intervalsModel.doc_topic_
    # doc_topic_max[doc] = topic
    doc_topic_max = np.argmax(doc_topic, axis=1)
    doc_topic_weight = np.max(doc_topic, axis=1)

    topic_word = intervalsModel.topic_word_
    topic_word_norm = topic_word / np.sum(topic_word, axis=0)
    n_top_words = 10

    topic_summary = ""
    topic_counts, _ = np.histogram(doc_topic_max, bins=numTopics)

    for ti in sorted(list(set(doc_topic_max))):
        tdocs = doc_topic_max == ti
        tdws = np.copy(doc_topic_weight)
        tdws[np.logical_not(tdocs)] = 0
        exemplar = np.argmax(tdws)
        mi = intervalMeetingIdx[exemplar]
        wmi = intervalWithinMeetingIdx[exemplar]

        t1 = wmi * interval
        t2 = t1 + interval
        print("Making transcript for topic {} with exemplar {} from meeting {} ({})".format(
            ti, exemplar, intervalMeetingName[exemplar], mi))
        transcript = makeTranscript(roots[mi], names[mi], start=t1, stop=t2)

        topic_words = vocab[np.argsort(topic_word_norm[ti, :])][: - (n_top_words + 1):-1]

        for tword in topic_words:
            transcript = transcript.replace(tword, "\u001b[31m" + tword + "\u001b[0m")
        out_file_str = '\n'.join([
            "Topic {}".format(ti),
            "Top words: {}".format(topic_words),
            "Exemplar: meeting {} ({}s-{}s) - weight {}".format(
                intervalMeetingName[exemplar], t1, t2, np.max(tdws)),
            transcript])

        output_file = os.path.join(output_dir, "interval_topic_{}.txt".format(ti))
        with open(output_file, "w") as f:
            f.write(out_file_str)

        topic_summary += "Topic {}, {} elements: {}\n".format(
            ti, topic_counts[ti], '; '.join(topic_words))

    output_file = os.path.join(output_dir, "interval_topic_summary.txt")
    with open(output_file, "w") as f:
        f.write(topic_summary)

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
