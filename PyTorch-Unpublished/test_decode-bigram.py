from decodeMulti_class_vocab_hidden_buckets_separated import GreedyDecoder
from statistics import mean 

decoder = GreedyDecoder("configs/config_E2E-origs2s-buckets-bigram-decode.json", top_n=1)
#GreedyDecoder("configs/config_E2E-hidden-buckets-decode.json",top_n = 1)

original_events = '/media/eilab/bd94e06c-afd2-4073-bb4f-a283b64dc505/Dropbox (GaTech)/Chicken/SciFiEventExperiments/DataSetup/Arg5/MiniSents-NameChange/Processed/all-sci-fi-justEvents.txt'


"""
events_from_file = [line.strip() for line in open("../DataSetup/Arg5/MiniSents-NameChange/E2Edata/5argSPEventsTest_input.txt").readlines()]

for line in events_from_file:
    event = str([line.split(" ")])
    #print("INPUT", event)
    print("OUTPUT",decoder.pipeline_predict(event))

"""


preds, golds, perplexity, bleus = decoder.test_set_eval_orig_bigram()#test_set_eval_orig() #test_set_eval()
print("AVERAGE TEST PERPLEXITY", perplexity)
print("AVERAGE TEST BLEU", mean(bleus))

for i, prediction in enumerate(preds):
    print("PREDICTED:", prediction)
    print("ORIGINAL:", golds[i])
    print("BLEU:",bleus[i])



