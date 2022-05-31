from decode_redo_pipeline import *
from statistics import mean 

#decoder = GreedyDecoder("configs/config_DRL-buckets-decode.json",top_n = 1)

#original_events = '/media/eilab/bd94e06c-afd2-4073-bb4f-a283b64dc505/Dropbox (GaTech)/Chicken/SciFiEventExperiments/DataSetup/Arg5/MiniSents-NameChange/Processed/all-sci-fi-justEvents.txt'


"""
#reading in events from a file
events_from_file = [line.strip() for line in open("../DataSetup/Arg5/MiniSents-NameChange/E2Edata/5argSPEventsTest_input.txt").readlines()]

for line in events_from_file:
	event = str([line.split(" ")])
	#print("INPUT", event)
	print("OUTPUT",decoder.pipeline_predict(event))

"""

"""
h = None
c = None
y= [['<ORGANIZATION>0', 'amuse-31.1', 'EmptyParameter', '<DATE>0', 'by']]
#y = ['<PERSON>10', 'give-13.1', "Synset('book.n.01')", '<PERSON>4', 'to']
output, _, _ = decoder.pipeline_predict(y, h, c, True)
print(output)
"""


"""
#numerically analyzing
preds, golds, perplexity, bleus = decoder.test_set_eval()
print("AVERAGE TEST PERPLEXITY", perplexity)
print("AVERAGE TEST BLEU", mean(bleus))

for i, prediction in enumerate(preds):
	print("PREDICTED:", prediction)
	print("ORIGINAL:", golds[i])
	print("BLEU:",bleus[i])


"""
decoder = Decoder("configs/config_DRL-buckets-decode.json")


events_from_file = [line.strip() for line in open("data/bucketed_events_test-DRL.txt").readlines()]

for line in events_from_file:
	if "<end_of_story>" in line: continue
	if "_of_story>" in line: continue
	event = [line.split(" ")]
	print("INPUT", event)
	output, _, _ = decoder.pipeline_predict(event, None, None, True)
	print("OUTPUT",output)
