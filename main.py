from PatchExtractor import PatchExtractor
from PIL import Image
from torchvision.transforms import Compose, PILToTensor
from PatchDataset import PatchDataset

dataSet = PatchDataset('data/set12/train/01.png', (5, 5))
first_elem, index = dataSet[0]
print(first_elem, first_elem.shape, len(dataSet))
#for patch in p_extractor:
#    count += 1
#    print(patch)
#print(count)

#dataset = PatchDataset('./set12', (3, 3))

#print(dataset[0])