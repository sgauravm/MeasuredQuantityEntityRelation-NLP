import pandas as pd
import os


class Preprocess:
    def __init__(self, tokenizer, entity_labels, label_type='BOI'):
        self.tokenizer = tokenizer
        self.entity_labels = entity_labels
        self.label_type = label_type
        label_symbols=None
        if label_type == 'BOI':
            label_symbols = ["B-", "I-"]
        id_to_lab = []
        for _, el in entity_labels.items():
            for b in label_symbols:
                id_to_lab.append(b + el)
        self.id_to_lab = id_to_lab.append('O')
        self.lab_to_id = lab_to_id = {lab: i for i, lab in enumerate(id_to_lab)}

    def BOI_labels(self, lab_map, offset_mapping, entity_labels):
        labels = [],
        label_ids = []
        cur = lab_map.pop()
        prev = False
        for token in offset_mapping:
            # Case:B - if first match
            if cur[1] == token[0]:
                labels.append("B-" + entity_labels[cur[0]])
                # if second doesnt match prev =True
                if cur[2] != token[1]:
                    prev = True
                else:
                    # pop label mappings
                    temp = lab_map.pop() if len(lab_map) != 0 else ['dummy', -1, -1]
                    while temp[1] == cur[1]:
                        cur = temp
                        temp = lab_map.pop() if len(lab_map) != 0 else ['dummy', -1, -1]
                    cur = temp
            # Case:I - if first doesnt match and prev==True
            elif cur[1] != token[0] and prev:
                labels.append("I-" + entity_labels[cur[0]])
                # if second matches, prev = False and pop label mappings
                if cur[2] == token[1]:
                    prev = False
                    # pop labels
                    temp = lab_map.pop() if len(lab_map) != 0 else ['dummy', -1, -1]
                    while temp[1] == cur[1]:
                        cur = temp
                        temp = lab_map.pop() if len(lab_map) != 0 else ['dummy', -1, -1]
                    cur = temp
            # Case:O - if first doesn't match and prev == False
            elif cur[1] != token[0] and not prev:
                labels.append("O")

        return labels

    # BOIES
    def BOIES_labels(self, lab_map, offset_mapping, entity_labels):
        labels = []
        cur = lab_map.pop()
        prev = False
        for token in offset_mapping:
            # If first and second both match, then 'S'
            if cur[1] == token[0] and cur[2] == token[1]:
                labels.append("S-" + entity_labels[cur[0]])
                # pop label
                temp = lab_map.pop() if len(lab_map) != 0 else ['dummy', -1, -1]
                while temp[1] == cur[1]:
                    cur = temp
                    temp = lab_map.pop() if len(lab_map) != 0 else ['dummy', -1, -1]
                cur = temp


            # Elseif first not match and prev==False, then 'O'
            elif cur[1] != token[0] and not prev:
                labels.append("O")

            # Elseif first match and second dont match,then 'B'
            elif cur[1] == token[0] and cur[2] != token[1]:
                labels.append("B-" + entity_labels[cur[0]])
                # set prev=True
                prev = True

            # Elseif first doesn't match,prev==True, and second doesn't match then 'I'
            elif cur[1] != token[0] and cur[2] != token[1] and prev:
                labels.append("I-" + entity_labels[cur[0]])

            # Elseif first doesn't match,prev==True, and second matches then 'E'
            elif cur[1] != token[0] and cur[2] == token[1] and prev:
                labels.append("E-" + entity_labels[cur[0]])
                # prev=False and pop label
                prev = False
                temp = lab_map.pop() if len(lab_map) != 0 else ['dummy', -1, -1]
                while temp[1] == cur[1]:
                    cur = temp
                    temp = lab_map.pop() if len(lab_map) != 0 else ['dummy', -1, -1]
                cur = temp

        return labels

    def read_files(self, path_text, path_tsv):
        all_files = os.listdir(path_text)
        texts = []
        text_labels = []
        exceptions = []
        for file in all_files:
            try:
                with open(os.path.join(path_text, file)) as f:
                    text = f.read()
                tsv_data = pd.read_csv(os.path.join(path_tsv, file.split('.')[0] + ".tsv"), sep="\t")[
                    ['annotType', 'startOffset', 'endOffset', 'text', 'annotId', 'other']].sort_values(by='startOffset',
                                                                                                       ascending=False)
                texts.append(text)
                text_labels.append(tsv_data.values.tolist())
            except Exception as e:
                exceptions.append(e)

        return texts, text_labels

    def get_encodings(self, path_text, path_tsv):
        texts, text_labels = self.read_files(path_text, path_tsv)
        encodings = self.tokenizer(texts, return_offsets_mapping=True, padding=True, truncation=True)
        labels = []
        label_fun = None
        if self.label_type=='BOI':
            label_fun = self.BOI_labels
        elif self.label_type=='BOIES':
            label_fun = self.BOIES_labels

        for offset_mapping, text_label in zip(encodings.offset_mappings, text_labels):
            label = label_fun(text_label, offset_mapping, self.entity_labels)
            labels.append(label)
        encodings['labels'] = [[self.lab_to_id[lab] for lab in item] for item in labels]
