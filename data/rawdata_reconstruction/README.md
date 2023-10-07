## Raw data reconstruction
This folder includes the link of raw data, codes and results of the reconstruction of the raw data. Please download raw data from the links, following their licenses.

### Raw data
- alpaca_gpt4: a public dataset released by [Flancuna Project](https://github.com/declare-lab/flacuna/tree/main/data)
- dialogsum: a public dataset released by [DialogSum Project](https://github.com/cylnlp/dialogsum/tree/main/DialogSum_Data)
- topiocqa: a public dataset released by [TopiocQA Project](https://mcgill-nlp.github.io/topiocqa/)

### Workflow
- run ```build_train.py``` and ```build_test.py``` to build the training and testing data in each dataset folder, respectively
- run ```mix_all_data.py``` to mix all the training data and testing data into single files
