from decode_forward_buckets import GreedyDecoder
from statistics import mean 

decoder = GreedyDecoder("configs/config_E2E-buckets-decode.json",top_n = 1)

#original_events = '/media/eilab/bd94e06c-afd2-4073-bb4f-a283b64dc505/Dropbox (GaTech)/Chicken/SciFiEventExperiments/DataSetup/Arg5/MiniSents-NameChange/Processed/all-sci-fi-justEvents.txt'



events_from_file = [line.strip() for line in open("../data/bucketed_events_test-DRL.txt").readlines()]

for line in events_from_file:
	if "<end_of_story>" in line: quit()
	if "_of_story>" in line: continue
	event = str([line.split(" ")])
	print("INPUT", event)
	print("OUTPUT",decoder.pipeline_predict(event))


"""
preds, golds, perplexity, bleus = decoder.test_set_eval()
print("AVERAGE TEST PERPLEXITY", perplexity)
print("AVERAGE TEST BLEU", mean(bleus))

for i, prediction in enumerate(preds):
    print("PREDICTED:", prediction)
    print("ORIGINAL:", golds[i])
    print("BLEU:",bleus[i])


"""
