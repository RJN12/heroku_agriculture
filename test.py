from keras.models import model_from_json
import scipy.io as sio
import numpy as np

classifier_model_weigts = "weights_proposal.mat"

def conv_dict(dict2):
    """Prepare the dictionary of weights to be loaded by the network
    :param dict2: Dictionary to format
    :returns: The dictionary properly formatted
    :rtype: dict
    """
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict
def pr(x):
    x = np.argmax(x)
    x
    if x == 0:
        return "healthy "
    elif x == 1:
        return "having anthracnose disease"
    elif x == 2 :
        return "having sooty mould disease"
    else:
        return "having powdermildew disease "
def load_weights(model, weights_file):
    """Loads the pretrained weights into the network architecture
    :param model: keras model of the network
    :param weights_file: Path to the weights file
    :returns: The input model with the weights properly loaded
    :rtype: keras.model
    """
    dict2 = sio.loadmat(weights_file)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def load_model(json_path):
    model = model_from_json(open(json_path).read())
    return model
def ld_model():
    model = load_model("model_proposal.json")
    model = load_weights(model, classifier_model_weigts)
    return model

raw = [261,329,332,381,240,199,185,6354]
norm = [float(i)/sum(raw) for i in raw]
model = ld_model()
d = model.predict(np.array(norm).reshape(1,8))
print(pr(d))
