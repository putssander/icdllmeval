from transformers import pipeline
import configparser


class NerMainPredict():

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('./../resources/config.ini')

        model = config["main"]["model"]
        print("load token-classification pipeline for main terms")

        self.token_classifier = pipeline(
            "token-classification", model=model, aggregation_strategy="simple"
        )

    def classify(self, txt):
        return self.token_classifier(txt)
    


main_ner =  NerMainPredict()
main_ner.classify("Mujer de 29 años con antecedentes de ulcus duodenal y estreñimiento que consulta por dolor en fosa renal derecha compatible con crisis renoureteral. No antecedentes de nefrolitiasis ni hematuria ni infecciones del tracto urinario. En la exploración sólo destaca una puñopercusión renal derecha positiva. La ecografía objetiva ectasia pielocalicial renal derecha con adelgazamiento del parénquima. La UIV muestra una anulación funcional de la unidad renal derecha siendo normal el resto de la exploración. Una pielografía retrograda muestra estenosis en la unión pieloureteral derecha, siendo la citología urinaria selectiva del uréter derecho negativa. Ante la disyuntiva de practicar una cirugía reconstructiva o una exerética se realiza gammagrafía renal que demuestra captación relativa del 33% para el riñón derecho y del 67% para el izquierdo.")