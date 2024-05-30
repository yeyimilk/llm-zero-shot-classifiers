

from transformers import pipeline
from llm_data_loader import get_test_data_by_name
from tqdm import tqdm
from utils.utils import save_to_file
from utils.report_utils import generate_reports

SMS_LABELS = {'normal': 0, 'spam': 1}
CORONA_LABELS = {'negative': 0, 'neutral': 1, 'positive': 2}
ECOMMERCE_LABELS = {'Household': 0, 'Books': 1, 'Clothing & Accessories': 2, 'Electronics': 3}
FINANCIAL_LABELS = {'positive': 2, 'neutral': 1, 'negative': 0}



def train(classifier, name, label_map):
    print(f"Start training {name}")

    x_test, y_test = get_test_data_by_name(name)
    labels = list(label_map.keys())

    results = []
    for x in tqdm(x_test, f'Predicting {name}...'):
        results.append(classifier(x, labels, multi_label=False))

    y_pred = [label_map[r['labels'][0]] for r in results]
    scores = [r['scores'] for r in results]
    reports = generate_reports(name, y_test, y_pred, scores)

    return {
        'reports': reports,
        'y_pred': y_pred,
        'y_test': y_test,
        'y_pred_d': scores
    }


def run_bart_deberta():
    bart_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    deberta_classifier = pipeline("zero-shot-classification", model="microsoft/deberta-large-mnli")
    model_names = ['bart', 'deberta']

    names = ['Corona_NLP_new', 'ecommerceDataset', 'sms_spam', 'financial_sentiment']
    label_maps = [CORONA_LABELS, ECOMMERCE_LABELS, SMS_LABELS, FINANCIAL_LABELS]
    classifiers = [bart_classifier, deberta_classifier]

    for i, name in tqdm(enumerate(names)):
        results = {}
        for j, classifier in enumerate(classifiers):
            result = train(classifier, name, label_maps[i])
            results[f'{model_names[j]}'] = result
        print(results)
        save_to_file(results, f'baseline_0shot/{name}', type='json')

if __name__ == '__main__':
    run_bart_deberta()