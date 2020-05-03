#Import data from file and store into two arrays
def import_data (path):
    with open (path, "r", newline='\n') as file:
        sentences, sentences_label, tokens, labels = [], [], [], []
        for line in file:
            # It wasn't possible to use csv method because some rows were corrupted
            line = line.strip().split("\t")

            if (len(line)>1):
                tokens.append(line[1])
                labels.append(line[2])
            elif(line[0] == ''):
                #There's a blank line: the sentence is over.
                sentences.append(tokens)
                sentences_label.append(labels)

                tokens, labels = [], []
        return sentences, sentences_label
