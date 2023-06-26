import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'./src'))
print(sys.path)

import unittest
from icdlmmeval.codiesp.codiformat import CodiFormat
from icdlmmeval import ner_parsing

class TestSum(unittest.TestCase):


    def test_single_term(self):
        codiformat = CodiFormat(path_codiesp='/home/jovyan/work/icdllmeval/notebooks/codiesp')
        df_train_x = codiformat.get_df_x("train")
        df_train_x.head(25)

        print(df_train_x.head(25))

        file = df_train_x["FILE"].tolist()[0]
        txt = codiformat.get_text("train", file)

        print(txt)
        sangrado = "varices"
        offsets = ner_parsing.get_offsets(txt, sangrado)
        offsets_s = ner_parsing.offset_to_string(offsets)

        self.assertEqual("1239 1246", offsets_s)


if __name__ == '__main__':
    unittest.main()