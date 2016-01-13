"""
the CTRobot class takes the full text of a clinical trial as input as a
string, and returns study characteristics predictions as a dict which can be
easily converted to JSON.

    text = "Streptomycin Treatment of Pulmonary Tuberculosis: A Medical Research Council Investigation..."

    robot = InterventionModelRobot()
    annotations = robot.annotate(text)

Models are trained on fields from clinicaltrials.gov. Note this annotator does
*no* annotation and simply predicts labels!

"""

# Authors:  Iain Marshall <mail@ijmarshall.com>
#           Joel Kuiper <me@joelkuiper.com>
#           Byron Wallce <byron.wallace@utexas.edu>


import codecs
import re
import pickle

from nltk.tokenize import sent_tokenize
from itertools import izip

from sklearn.feature_extraction.text import HashingVectorizer

FIRST_N = 20 # number of sentences to consider as the abstract


class CTRobot:
    '''
    Class for making predicton of study characteristics from abstracts

    '''
    study_chars = ('Allocation', 'Endpoint Classification', 'Intervention Model', 'Masking', 'Primary Purpose', 'Gender', 'Healthy Volunteers', 'Phase')

    def __init__(self):
        '''
        Initialize vectorizer and clfs

        '''
        self.vec = HashingVectorizer(ngram_range=(1, 2), stop_words='english', binary=True)

        self.clfs = [self._load_model(study_char) for study_char in CTRobot.study_chars]

    @staticmethod
    def _load_model(study_char):
        '''
        Unpickle the classifier from disk

        '''
        study_char = re.sub('\s+', '_', study_char.lower())

        return pickle.load(open('robots/study_chars/{}_clf.p'.format(study_char), 'rb'))

    def annotate(self, doc_text):
        '''
        Predict study characteristics from abstract of `doc_text`

        '''
        abstract = ' '.join(sent_tokenize(doc_text)[:FIRST_N]) # approximate abstract as first N sentences
        X = self.vec.transform([abstract])

        # Predictions for each study characteristic
        preds = {study_char: clf.predict(X)[0] for study_char, clf in izip(CTRobot.study_chars, self.clfs)}

        # Markdownify
        markdowns = ['**{}**: {}'.format(study_char, pred) for study_char, pred in preds.items()]
        markdown = '\n\n'.join(markdowns) # double newlines markdown specific

        marginalia = {
            "type": "Study Characteristics",
            "title": "Study Characteristics",
            "description": markdown
        }

        return {"marginalia": [marginalia]}

def main():
    # Read in example input to the text string
    with codecs.open('tests/example.txt', 'r', 'ISO-8859-1') as f:
        text = f.read()

    lines = text.split('\n')

    # make a Clinical Trials robot, use it to make predictions
    robot = CTRobot()
    annotations = robot.annotate(' '.join(lines))

    print "EXAMPLE OUTPUT:"
    print
    print annotations


if __name__ == '__main__':
    main()
