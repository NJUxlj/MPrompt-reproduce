import copy
import json
import rouge
import logging


logger = logging.getLogger(__name__)


rouge_l_evaluator = rouge.Rouge(
    metrics = ["rouge-l"]
)



def rouge_l(prediction,gold):
    try:
        return rouge_l_evaluator.get_scores(prediction, gold, avg=True)
    except:
        logger.info(f"error rouge-l predict: {prediction}")
        return {
                'rouge-l': {
                    'r': 0.0, 
                    'p': 0.0, 
                    'f': 0.0
                }   
            }



def metric_max_over_ground_truth(metric_fn, prediction, ground_truths):
    pass




def get_rouge_l(predictions, golds):
    pass
