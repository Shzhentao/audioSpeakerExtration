import os
from pyannote.audio import Inference
from typing import DefaultDict
import numpy as np
import pickle

if __name__ == "__main__":
    inference = Inference("pyannote/embedding", window="whole")
    wsj0Root = "/media/aiden/D83891573891358A/datasets/wsj0-mix/2speakers/wav8k/min"
    featureDict = DefaultDict()
    fp = open("./wsj_vp.pkl", "wb")
    for path, dirL, fileL in os.walk(wsj0Root):
        for file in fileL:
            filesuffix = file.split(".")[-1]
            if filesuffix == "wav":
                lastdirname = path.split("/")[-1]
                if not lastdirname == "mix":
                    feature = inference(os.path.join(path, file))
                    newfeature = feature[np.newaxis, :]
                    # "/".join(os.path.join(path, file).split("/")[-3:])
                    featureDict[file] = newfeature
    pickle.dump(featureDict, fp)
                